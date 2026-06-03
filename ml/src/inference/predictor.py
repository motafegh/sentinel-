"""
predictor.py — SENTINEL Inference Predictor (Cross-Attention + LoRA Upgrade)

WHAT CHANGED FROM TRACK 3 ORIGINAL:
    1. Reads architecture type from checkpoint config
       "cross_attention_lora" → SentinelModel with fusion_output_dim=128
       missing/other          → falls back to 64 (backward-compat with old checkpoints)

    2. fusion_output_dim passed to SentinelModel from checkpoint config
       Prevents silent shape mismatch when loading old vs new checkpoints.

    3. weights_only=False kept for checkpoint loading
       LoRA state dict contains peft-specific classes — weights_only=True blocks them.
       (Graph/token datasets now use weights_only=True, fixed in dual_path_dataset.py)

    4. Per-class thresholds loaded from {checkpoint.stem}_thresholds.json
       Replaces single global threshold with class-specific decision boundaries.
       If file missing, falls back to user-supplied threshold (or 0.5).

    5. Stores self.architecture for health endpoint (avoid reloading checkpoint)

FIXES (2026-04-29):
    Bug 4 — Unknown architecture now raises ValueError (was silent fallthrough to 64).
             _ARCH_TO_FUSION_DIM allowlist replaces if/else.
    Bug 5 — num_classes > len(CLASS_NAMES) now raises ValueError before slicing.
             Prevents zip() silent truncation if a future checkpoint adds new classes.
    Bug 3 — _score() emits 'vulnerability_class' key (was 'class').
             predictor.py is the canonical schema owner; no consumer remapping needed.

FIXES (2026-05-01):
    Audit #5 — _warmup() uses a 2-node 1-edge graph so GATConv.propagate() is
               called during startup. A 0-edge graph skips message-passing entirely,
               hiding GAT shape bugs until the first real inference request.

FIXES (2026-05-04):
    Fix #2 — SentinelModel() now reads dropout (fusion_dropout), gnn_dropout, and
               lora_target_modules from saved checkpoint config. Previously missing
               args caused load_state_dict() crash when checkpoint used non-defaults.
    Fix #4 — _warmup() dummy graph now includes edge_attr when use_edge_attr=True,
               exercising the full nn.Embedding code path at startup.
    Fix #6 — _format_result() now returns "thresholds": self.thresholds.cpu().tolist()
               (a per-class list) instead of "threshold": self.threshold (single float).
               BREAKING CHANGE: API consumers must update to read 'thresholds' (list).
    Fix #7 — fusion_output_dim lookup now prefers saved_cfg.get("fusion_output_dim")
               and only falls back to _ARCH_TO_FUSION_DIM for legacy checkpoints that
               predate trainer.py saving the value directly into the checkpoint config.

IMPROVEMENTS:
    - self.thresholds_loaded flag exposed for /health endpoint.
    - Warns per-class when threshold JSON is missing a class entry (uses fallback silently).
    - Strict checkpoint metadata validation: fusion_output_dim and class_names in config
      are cross-checked when present, catching stale checkpoints early.
    - Warmup forward pass at startup catches CUDA / model-shape issues before first request.
    - Legacy binary mode (legacy_binary) logs an explicit warning so production operators
      notice an accidental legacy checkpoint load.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger
from torch_geometric.data import Batch, Data

from ml.src.inference.preprocess import ContractPreprocessor
from ml.src.models.sentinel_model import (
    SentinelModel,
    _FUNC_TYPE_IDS,   # frozenset of function-level node type IDs
    _MAX_TYPE_ID,     # float normalisation constant (max NODE_TYPES value)
)
from ml.src.preprocessing.graph_schema import NODE_FEATURE_DIM, NODE_TYPES
from ml.src.training.trainer import CLASS_NAMES

# Plain Python set for fast membership checks in hotspot extraction
# (avoids torch.isin overhead for small per-node iteration).
_FUNC_TYPE_IDS_SET: frozenset[int] = _FUNC_TYPE_IDS

# Must match retokenize_windowed.py MAX_WINDOWS=4.
# Inference batches exactly this many windows (padding with zeros when W < 4)
# so WindowAttentionPooler and CrossAttentionFusion see the same [B, W*512, 768]
# context they were trained on.  Changing this without retraining will break calibration.
_TRAINING_MAX_WINDOWS: int = 4

_BINARY_CLASS_NAME = "BinaryScore"


def _ensure_list(v: object) -> list:
    """Guard: MLflow may serialise list[str] as a comma-joined string."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [s.strip() for s in v.split(",")]
    return list(v)

# ---------------------------------------------------------------------------
# Bug 4 fix — explicit architecture allowlist (replaces silent else-64 branch)
# ---------------------------------------------------------------------------
# Registry pattern: adding a new architecture = one dict entry, not an elif hunt.
# Keys must match exactly what trainer.py writes into checkpoint["config"]["architecture"].
_ARCH_TO_FUSION_DIM: dict[str, int] = {
    "four_eye_v8":          128,   # Run7+ four-eye: GNN+TF+Fused+CFG, type embedding, Phase2 heads=4
    "three_eye_v8":         128,   # v8 three-eye classifier — 8-layer GNN, gnn_prefix_k support
    "three_eye_v7":         128,   # v7 three-eye classifier — 11-dim nodes, 7-layer GNN
    "three_eye_v5":         128,   # v5 three-eye classifier
    "cross_attention_lora": 128,   # v4 (previous)
    "legacy":               64,
    "legacy_binary":        64,
}

# Node feature dimension per architecture — used for warmup dummy graph.
# Current architecture imports directly from graph_schema; legacy values are hardcoded.
_ARCH_TO_NODE_DIM: dict[str, int] = {
    "four_eye_v8":          NODE_FEATURE_DIM,  # Run7+ — same stored schema, type embedding is model-internal
    "three_eye_v8":         NODE_FEATURE_DIM,  # v8 — same schema as v7
    "three_eye_v7":         NODE_FEATURE_DIM,  # always in sync with schema
    "three_eye_v5":         NODE_FEATURE_DIM,  # always in sync with schema
    "cross_attention_lora": 8,     # v4 legacy
    "legacy":               8,
    "legacy_binary":        8,
}


class Predictor:
    """
    Loads a trained SentinelModel checkpoint and scores Solidity contracts.

    Handles both architectures:
        cross_attention_lora: fusion_output_dim=128, CrossAttentionFusion
        legacy (concat+MLP):  fusion_output_dim=64,  FusionLayer (old)

    Per‑class thresholds are read from a JSON file named
    `{checkpoint.stem}_thresholds.json` located next to the checkpoint.
    If the file is missing, the single `threshold` argument is used for all classes.

    Args:
        checkpoint: Path to .pt checkpoint (new dict format).
        threshold:  Fallback threshold for all classes when no per‑class file exists.
                    Default 0.50.
        device:     Auto-detected if not supplied.
    """

    DEFAULT_THRESHOLD = 0.50

    # Three-tier suspicion output thresholds.
    # CONFIRMED  — model confident enough to commit; agents hard-flag + ZK proof candidate.
    # SUSPICIOUS — non-trivial signal; agents send to RAG + static analysis for verification.
    # Classes below SUSPICIOUS_THRESHOLD are NOTEWORTHY (included in probabilities dict only).
    TIER_CONFIRMED_THRESHOLD:  float = 0.55
    TIER_SUSPICIOUS_THRESHOLD: float = 0.25

    def __init__(
        self,
        checkpoint: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
        tier_confirmed_threshold:  float | None = None,
        tier_suspicious_threshold: float | None = None,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

        self.threshold = threshold  # kept for backward compat; not used in _score
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tier thresholds — caller can override class defaults for deployment tuning.
        self.tier_confirmed_threshold  = tier_confirmed_threshold  or self.TIER_CONFIRMED_THRESHOLD
        self.tier_suspicious_threshold = tier_suspicious_threshold or self.TIER_SUSPICIOUS_THRESHOLD

        logger.info(f"Predictor initialising on: {self.device}")
        logger.info(f"Checkpoint: {checkpoint}")

        # ------------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------------
        raw = torch.load(checkpoint, map_location=self.device, weights_only=False)

        if isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
            saved_cfg = raw.get("config", {})
            num_classes = saved_cfg.get("num_classes", len(CLASS_NAMES))
            architecture = saved_cfg.get("architecture", "legacy")

            # Bug 4 fix — reject unknown architecture immediately.
            if architecture not in _ARCH_TO_FUSION_DIM:
                raise ValueError(
                    f"Unknown checkpoint architecture: '{architecture}'. "
                    f"Known architectures: {list(_ARCH_TO_FUSION_DIM.keys())}. "
                    "Add the new architecture to _ARCH_TO_FUSION_DIM in predictor.py "
                    "and set the correct fusion_output_dim before loading this checkpoint."
                )

            # Fix #7: prefer fusion_output_dim stored directly in config;
            # fall back to _ARCH_TO_FUSION_DIM only for legacy checkpoints
            # that predate trainer.py saving this value into the config dict.
            fusion_output_dim = saved_cfg.get(
                "fusion_output_dim", _ARCH_TO_FUSION_DIM.get(architecture, 128)
            )

            # Strict metadata cross-check: class_names order
            cfg_class_names = saved_cfg.get("class_names")
            if cfg_class_names is not None:
                expected = CLASS_NAMES[:len(cfg_class_names)]
                if cfg_class_names != expected:
                    raise ValueError(
                        f"Checkpoint class_names mismatch.\n"
                        f"  Checkpoint: {cfg_class_names}\n"
                        f"  Expected:   {expected}\n"
                        "CLASS_NAMES must not be reordered or renamed. "
                        "Only append new classes at the end."
                    )

            logger.info(
                f"Checkpoint — epoch: {raw.get('epoch', '?')} | "
                f"best_f1: {raw.get('best_f1', 0):.4f} | "
                f"num_classes: {num_classes} | "
                f"architecture: {architecture} | "
                f"fusion_dim: {fusion_output_dim}"
            )
        else:
            # Very old binary checkpoint — plain state_dict
            state_dict = raw
            num_classes = 1
            fusion_output_dim = 64
            architecture = "legacy_binary"
            saved_cfg = {}
            logger.warning(
                "Old-format checkpoint — loading as binary (num_classes=1). "
                "This checkpoint predates Track 3 multi-label mode. "
                "Do NOT use in production without explicit intent."
            )

        # Bug 5 fix — guard before slice so zip() never silently drops classes.
        if num_classes > len(CLASS_NAMES):
            raise ValueError(
                f"Checkpoint num_classes={num_classes} exceeds "
                f"CLASS_NAMES length={len(CLASS_NAMES)}. "
                f"Append the new class name(s) to CLASS_NAMES in "
                f"ml/src/training/trainer.py before loading this checkpoint."
            )

        self.num_classes = num_classes
        self.architecture = architecture          # stored for health endpoint
        self._class_names = CLASS_NAMES[:num_classes] if num_classes > 1 else [_BINARY_CLASS_NAME]
        self._saved_cfg = saved_cfg              # kept for _warmup()

        # ------------------------------------------------------------------
        # Build model with correct architecture
        # Fix #2: pass dropout, gnn_dropout, lora_target_modules from checkpoint
        # config so load_state_dict() never crashes due to LoRA key mismatches.
        # ------------------------------------------------------------------
        # Architecture-aware defaults: v5 uses wider GNN + higher LoRA rank.
        _is_v5 = (architecture == "three_eye_v5")
        self.model = SentinelModel(
            num_classes=num_classes,
            fusion_output_dim=fusion_output_dim,
            gnn_hidden_dim=saved_cfg.get("gnn_hidden_dim", 128 if _is_v5 else 64),
            gnn_num_layers=saved_cfg.get("gnn_layers", 4),
            gnn_heads=saved_cfg.get("gnn_heads", 8),
            use_edge_attr=saved_cfg.get("use_edge_attr", True),
            gnn_edge_emb_dim=saved_cfg.get("gnn_edge_emb_dim", 32 if _is_v5 else 16),
            gnn_use_jk=saved_cfg.get("gnn_use_jk", _is_v5),
            lora_r=saved_cfg.get("lora_r", 16 if _is_v5 else 8),
            lora_alpha=saved_cfg.get("lora_alpha", 32 if _is_v5 else 16),
            lora_dropout=saved_cfg.get("lora_dropout", 0.1),
            dropout=saved_cfg.get("fusion_dropout", 0.3),
            gnn_dropout=saved_cfg.get("gnn_dropout", 0.2),
            lora_target_modules=_ensure_list(
                saved_cfg.get("lora_target_modules", ["query", "value"])
            ),
            gnn_prefix_k=saved_cfg.get("gnn_prefix_k", 0),
            gnn_prefix_warmup_epochs=saved_cfg.get("gnn_prefix_warmup_epochs", 15),
        ).to(self.device)
        # Strip _orig_mod. prefix left by torch.compile when saving compiled checkpoints
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        # Resize edge_embedding if checkpoint used fewer edge types than current schema
        edge_emb_key = next((k for k in state_dict if "edge_embedding.weight" in k), None)
        if edge_emb_key and self.model.gnn.edge_embedding is not None:
            ckpt_num_edge_types = state_dict[edge_emb_key].shape[0]
            current = self.model.gnn.edge_embedding.num_embeddings
            if ckpt_num_edge_types != current:
                import torch.nn as nn
                emb_dim = self.model.gnn.edge_embedding.embedding_dim
                self.model.gnn.edge_embedding = nn.Embedding(ckpt_num_edge_types, emb_dim).to(self.device)
                logger.info(f"Resized edge_embedding: {current} → {ckpt_num_edge_types} types (checkpoint predates current schema)")
        self.model.load_state_dict(state_dict)
        self.model.float()  # Normalize BF16 AMP checkpoints to float32 for inference
        self.model.eval()
        self.model._current_epoch = 9999  # prefix always active at inference (no warmup suppression)
        self.model.parameter_summary()

        # ------------------------------------------------------------------
        # Load per‑class thresholds from companion JSON (if exists)
        # ------------------------------------------------------------------
        checkpoint_path = Path(checkpoint)
        thresholds_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_thresholds.json")

        if thresholds_path.exists():
            with thresholds_path.open("r") as f:
                thresholds_data = json.load(f)
            class_thresholds_dict = thresholds_data.get("thresholds", {})

            per_class_thresholds = []
            missing_classes = []
            for cls_name in self._class_names:
                if cls_name in class_thresholds_dict:
                    per_class_thresholds.append(class_thresholds_dict[cls_name])
                else:
                    per_class_thresholds.append(self.threshold)
                    missing_classes.append(cls_name)

            if missing_classes:
                logger.warning(
                    f"Threshold JSON missing entries for: {missing_classes} — "
                    f"using fallback threshold {self.threshold} for these classes. "
                    "Re-run tune_threshold.py to generate complete per-class thresholds."
                )

            self.thresholds = torch.tensor(
                per_class_thresholds, dtype=torch.float32, device=self.device
            )
            self.thresholds_loaded = True
            logger.info(
                f"Loaded per‑class thresholds from {thresholds_path} — "
                f"min={self.thresholds.min().item():.3f}, max={self.thresholds.max().item():.3f}"
            )
        else:
            self.thresholds = torch.full(
                (self.num_classes,), self.threshold, dtype=torch.float32, device=self.device
            )
            self.thresholds_loaded = False
            logger.warning(
                f"No thresholds JSON found at {thresholds_path} — "
                f"using uniform threshold {self.threshold} for all classes"
            )

        # ------------------------------------------------------------------
        # Preprocessor — loaded once, reused per call
        # ------------------------------------------------------------------
        self.preprocessor = ContractPreprocessor()

        # ------------------------------------------------------------------
        # Warmup forward pass — catches CUDA / model-shape issues at startup,
        # not on the first real request.
        # ------------------------------------------------------------------
        self._warmup()
        logger.info(f"Predictor ready | {self.num_classes} classes | {architecture}")

    def _warmup(self) -> None:
        """
        Run one minimal forward pass with dummy tensors to surface CUDA and
        model-shape issues at startup instead of on the first real request.

        Graph dummy (3 nodes, exercising all critical code paths):
          node 0 — CONTRACT    : graph root, always present
          node 1 — FUNCTION    : triggers select_prefix_nodes() → prefix injection
          node 2 — STATE_VAR   : non-function leaf; tests type_id normalisation
          edges  — 0↔1 CALLS, 1→2 CONTAINS (undirected where needed)
          edge_attr — valid Embedding indices for the active edge types

        Rationale for FUNCTION node:
          select_prefix_nodes() only selects FUNCTION/MODIFIER/FALLBACK/RECEIVE/
          CONSTRUCTOR nodes.  The previous 2-node graph (both STATE_VAR, type_id=0)
          yielded 0 eligible nodes, so gnn_to_bert_proj and prefix_type_embedding
          were never executed during warmup.  A bug in those layers would survive
          startup and surface only on the first real contract.  Adding one FUNCTION
          node forces the full prefix code path to run.

        Token dummy ([1, _TRAINING_MAX_WINDOWS, 512] batched format):
          Matches the [B, W, L] shape that training sends.  This exercises
          WindowAttentionPooler's multi-window learned-attention path (WL=2048 > 512)
          and CrossAttentionFusion's 2048-position key/value context — both of which
          are bypassed when a single [1, 512] tensor is passed instead.
          Windows 1-3 are all-zero (mask=0) replicating how offline pads short contracts.
        """
        try:
            _node_dim = _ARCH_TO_NODE_DIM.get(self.architecture, NODE_FEATURE_DIM)
            _max_type = float(max(NODE_TYPES.values()))   # 12.0 for v8

            # ── dummy graph: CONTRACT(0) + FUNCTION(1) + STATE_VAR(2) ──────────
            dummy_x = torch.zeros(3, _node_dim, dtype=torch.float32, device=self.device)
            # dim[0] = type_id / max_type_id  (normalised node type)
            dummy_x[0, 0] = NODE_TYPES.get("CONTRACT",   7) / _max_type
            dummy_x[1, 0] = NODE_TYPES.get("FUNCTION",   1) / _max_type
            dummy_x[2, 0] = NODE_TYPES.get("STATE_VAR",  0) / _max_type

            # Edges: 0↔1 (CALLS, bidirectional) + 1→2 (CONTAINS)
            dummy_edge_index = torch.tensor(
                [[0, 1, 1], [1, 0, 2]], dtype=torch.long, device=self.device
            )  # [2, 3]

            use_edge_attr = self._saved_cfg.get("use_edge_attr", True)
            dummy_kwargs: dict = dict(x=dummy_x, edge_index=dummy_edge_index)
            if use_edge_attr:
                from ml.src.preprocessing.graph_schema import EDGE_TYPES
                calls_id    = EDGE_TYPES.get("CALLS",    0)
                contains_id = EDGE_TYPES.get("CONTAINS", 5)
                dummy_kwargs["edge_attr"] = torch.tensor(
                    [calls_id, calls_id, contains_id], dtype=torch.long, device=self.device
                )  # [E=3]

            dummy_graph = Data(**dummy_kwargs)
            dummy_batch = Batch.from_data_list([dummy_graph]).to(self.device)

            # ── token dummy: [1, W, 512] — batched multi-window format ──────────
            # Window 0: first token real (avoids empty masked-mean in attention)
            # Windows 1-3: all PAD (mask=0) — replicates offline zero-padding of short contracts
            W   = _TRAINING_MAX_WINDOWS
            dummy_ids  = torch.zeros(1, W, 512, dtype=torch.long,  device=self.device)
            dummy_mask = torch.zeros(1, W, 512, dtype=torch.long,  device=self.device)
            dummy_mask[0, 0, 0] = 1   # one real token in window 0

            with torch.no_grad():
                _ = self.model(dummy_batch, dummy_ids, dummy_mask)

            logger.info(
                f"Warmup forward pass succeeded — model ready "
                f"(3-node graph with FUNCTION: prefix path exercised; "
                f"[1,{W},512] token format: WindowAttentionPooler multi-window path exercised)"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Model warmup failed — checkpoint may be incompatible with current code. "
                f"Error: {exc}"
            ) from exc

    def predict_with_hotspots(self, source_code: str) -> dict:
        """
        Run full inference AND return per-function GNN attention hotspots.

        Hotspot scoring uses the L2 norm of each function-level node's GNN embedding
        as a proxy for how much structural signal the GNN concentrated on that node.
        GAT layers route message-passing signal proportionally to attention weights;
        nodes that aggregate more neighbourhood signal end up with higher-norm embeddings.

        Function → score mapping:
          - Only FUNCTION/MODIFIER/FALLBACK/RECEIVE/CONSTRUCTOR nodes are scored
            (same filter used by select_prefix_nodes).
          - Score = mean L2 norm of the node's GNN embedding, normalised to [0, 1]
            across all function nodes in the contract.
          - Lines come from node_metadata["source_lines"] stored by graph_extractor.
          - Node IDs are the raw PyG node indices (0-indexed, stable across calls).

        Returns the standard predict_source() result dict plus:
          "hotspots": [
            {
              "fn_name":    str,          # canonical_name from Slither AST
              "node_id":    int,          # PyG node index
              "score":      float,        # normalised [0, 1]
              "lines":      list[int],    # source line numbers (may be empty)
              "node_type":  str,          # FUNCTION | MODIFIER | FALLBACK | etc.
            }, ...                        # sorted descending by score, top 20
          ],
          "hotspot_stats": {
            "total_function_nodes": int,
            "num_nodes":            int,
            "attention_source":     "gnn_embedding_norm"
          }
        """
        graph, windows = self.preprocessor.process_source_windowed(source_code)
        result = self._score_windowed(graph, windows)
        result["hotspots"] = []
        result["hotspot_stats"] = {
            "total_function_nodes": 0,
            "num_nodes": int(graph.num_nodes),
            "attention_source": "gnn_embedding_norm",
        }

        if self.architecture not in ("four_eye_v8", "three_eye_v8", "three_eye_v7", "three_eye_v5"):
            # Legacy architectures don't have the GNN structure needed
            return result

        if not hasattr(graph, "node_metadata") or graph.node_metadata is None:
            return result

        try:
            self.model.eval()
            batch = Batch.from_data_list([graph]).to(self.device)
            edge_attr = getattr(batch, "edge_attr", None) if self._saved_cfg.get("use_edge_attr", True) else None

            with torch.no_grad():
                # Run only the GNN encoder to get per-node embeddings
                node_embs, _, _ = self.model.gnn(
                    batch.x, batch.edge_index, batch.batch, edge_attr
                )  # [N, gnn_hidden_dim]

            node_embs_cpu = node_embs.float().cpu()  # detach from autograd + ensure float32

            # Recover integer type IDs from the normalised feature dim[0]
            node_type_ids = (batch.x[:, 0].float().cpu() * _MAX_TYPE_ID).round().long()

            # Find function-level nodes (same filter as select_prefix_nodes)
            func_indices = [
                i for i, tid in enumerate(node_type_ids.tolist())
                if tid in _FUNC_TYPE_IDS_SET
            ]

            if not func_indices:
                return result

            # Compute L2 norm of each function node's embedding as hotspot score
            scores_raw = [
                float(node_embs_cpu[i].norm(p=2).item())
                for i in func_indices
            ]

            # Normalise to [0, 1]; guard against all-equal case
            min_s, max_s = min(scores_raw), max(scores_raw)
            span = max_s - min_s if max_s > min_s else 1.0
            scores_norm = [(s - min_s) / span for s in scores_raw]

            metadata = graph.node_metadata
            type_name_map = {v: k for k, v in NODE_TYPES.items()}

            hotspots = []
            for rank, (node_idx, norm_score) in enumerate(
                sorted(zip(func_indices, scores_norm), key=lambda x: -x[1])
            ):
                if rank >= 20:  # return top-20 only
                    break
                meta = metadata[node_idx] if node_idx < len(metadata) else {}
                type_id = int(node_type_ids[node_idx].item())
                hotspots.append({
                    "fn_name":   meta.get("name", f"node_{node_idx}"),
                    "node_id":   node_idx,
                    "score":     round(norm_score, 4),
                    "lines":     meta.get("source_lines", []),
                    "node_type": type_name_map.get(type_id, "FUNCTION"),
                })

            result["hotspots"] = hotspots
            result["hotspot_stats"] = {
                "total_function_nodes": len(func_indices),
                "num_nodes":            int(graph.num_nodes),
                "attention_source":     "gnn_embedding_norm",
            }

        except Exception as exc:
            logger.warning(f"Hotspot extraction failed (non-fatal): {exc}")
            # Return base predict result without hotspots rather than raising

        return result

    def predict(self, sol_path: str | Path) -> dict:
        """
        Score a Solidity contract file on disk.

        Reads the source text and delegates to predict_source() so that:
          - Sliding-window tokenisation is used for long contracts (no silent truncation)
          - The forward pass uses the same [1, W, 512] batched format as training
          - WindowAttentionPooler's learned attention weights are exercised

        Previously this called preprocessor.process() → single [1, 512] window, which
        silently truncated contracts longer than 512 tokens and sent a shape that bypasses
        WindowAttentionPooler's multi-window path (DISCREPANCY-5 / audit O3).
        """
        source = Path(sol_path).read_text(encoding="utf-8", errors="replace")
        return self.predict_source(source)

    def predict_source(self, source_code: str, name: str = "contract") -> dict:
        """
        Score a raw Solidity source string using the training-aligned forward path.

        Always uses the batched multi-window format [1, _TRAINING_MAX_WINDOWS, 512]:
          - Short contracts (≤ 510 content tokens): 1 real window + 3 zero-padded
          - Long contracts: up to _TRAINING_MAX_WINDOWS=4 sliding windows

        This ensures:
          1. WindowAttentionPooler uses its learned attention (WL=2048 > 512 triggers
             the multi-window path instead of the single-window CLS fallback).
          2. CrossAttentionFusion sees 4×512=2048 token positions — identical to training.
          3. No source truncation: the sliding window covers up to 4×510=2040 content
             tokens (vs 510 max with a single window).
        """
        graph, windows = self.preprocessor.process_source_windowed(source_code)
        return self._score_windowed(graph, windows)

    def _score_windowed(self, graph, windows: list[dict]) -> dict:
        """
        Single batched forward pass over all windows — training-aligned.

        Training path (DualPathDataset + dual_path_collate_fn):
          offline .pt tensor: [W, 512] → collated to [B, W, 512]
          model receives:     [B, W, 512]  where W = MAX_WINDOWS = 4
          TransformerEncoder: flattens to [B*W, 512] → CodeBERT → [B, W*512, 768]
          WindowAttentionPooler: WL=2048 > 512 → learned-attention over 4 CLS vectors
          CrossAttentionFusion:  key/value [B, 2048, 768]

        Inference path (this method):
          windows list (1–4 real, from process_source_windowed)
          → capped at _TRAINING_MAX_WINDOWS=4 (model never trained with W>4)
          → padded with zero windows to exactly W=4 (mask=0 → fusion ignores them,
            matching how offline pads short contracts in retokenize_windowed.py)
          → stacked to [1, 4, 512]
          → single model() call — identical shape to training
          → WindowAttentionPooler multi-window path, CrossAttentionFusion 2048-context

        This replaces the old per-window loop + max-pool which:
          - Sent [1, 512] per window → WL=512 ≤ 512 → WindowAttentionPooler fallback
            (learned attention bypassed, raw CLS used instead)
          - Ran W separate forward passes (graph re-processed W times unnecessarily)
          - Max-pooled sigmoid probabilities (different aggregation from training)
        """
        self.model.eval()
        batch = Batch.from_data_list([graph]).to(self.device)

        n_real = len(windows)

        # Cap: model calibrated for W=4; extra windows would shift CLS positions
        # in WindowAttentionPooler beyond what the learned scorer expects.
        selected = windows[:_TRAINING_MAX_WINDOWS]

        # Pad to exactly _TRAINING_MAX_WINDOWS with all-zero windows.
        # attention_mask=0 on pad windows → CrossAttentionFusion key_padding_mask
        # masks them out (same mechanism as offline zero-padded windows).
        pad_ids  = torch.zeros(1, 512, dtype=torch.long, device=self.device)
        pad_mask = torch.zeros(1, 512, dtype=torch.long, device=self.device)
        padded = list(selected)
        while len(padded) < _TRAINING_MAX_WINDOWS:
            padded.append({"input_ids": pad_ids, "attention_mask": pad_mask})

        # Stack: [W, 1, 512] → cat on dim=0 → [W, 512] → unsqueeze → [1, W, 512]
        stacked_ids  = torch.cat(
            [w["input_ids"].to(self.device)      for w in padded], dim=0
        ).unsqueeze(0)   # [1, W, 512]
        stacked_mask = torch.cat(
            [w["attention_mask"].to(self.device) for w in padded], dim=0
        ).unsqueeze(0)   # [1, W, 512]

        with torch.no_grad():
            logits = self.model(batch, stacked_ids, stacked_mask)   # [1, num_classes]

        probs = torch.sigmoid(logits.float()).squeeze(0)   # [num_classes]
        return self._format_result(graph, probs, windows[0], n_real)

    def _score(self, graph, tokens: dict) -> dict:
        """
        Run forward pass for a single token window and return structured result.

        Sigmoid applied here — NOT inside model (BCEWithLogitsLoss compatibility).
        Per‑class thresholds are applied instead of a single global threshold.
        """
        self.model.eval()

        with torch.no_grad():
            batch = Batch.from_data_list([graph]).to(self.device)
            input_ids = tokens["input_ids"].to(self.device)       # [1, 512]
            attention_mask = tokens["attention_mask"].to(self.device)  # [1, 512]

            logits = self.model(batch, input_ids, attention_mask)  # [1, num_classes]
            probs = torch.sigmoid(logits.float()).squeeze(0)       # [num_classes]

        return self._format_result(graph, probs, tokens, windows_used=1)

    def _format_result(
        self,
        graph,
        probs: torch.Tensor,   # [num_classes] CPU or GPU
        tokens: dict,
        windows_used: int,
    ) -> dict:
        """
        Convert probability tensor + metadata into the three-tier result dict.

        Three-tier suspicion output design (2026-05-27):
          CONFIRMED  (prob >= tier_confirmed_threshold=0.55):
            Model is confident. Hard-flag for agents; ZK proof candidate.
          SUSPICIOUS (tier_suspicious_threshold=0.25 <= prob < 0.55):
            Non-trivial signal. Send to RAG + static analysis for verification.
            Evidence from 20-contract evaluation: 8 missed classes had prob
            0.25-0.54 — binary threshold was discarding real signal.
          NOTEWORTHY (prob < 0.25):
            Weak signal. Included in probabilities dict only; not in tier lists.

        Schema:
          label           "safe" | "suspicious" | "confirmed_vulnerable"
          probabilities   {class: float} — full 10-class vector, ALWAYS present
          confirmed       [{vulnerability_class, probability, tier="CONFIRMED"}, ...]
          suspicious      [{vulnerability_class, probability, tier="SUSPICIOUS"}, ...]
          vulnerabilities legacy alias for confirmed (backward compat — was above-threshold)
          tier_thresholds {"confirmed": 0.55, "suspicious": 0.25, "noteworthy": 0.10}
          thresholds      [float × num_classes] — per-class tuned thresholds (unchanged)
          truncated       bool
          windows_used    int
          num_nodes       int
          num_edges       int
        """
        probs_cpu = probs.cpu()
        probs_list: list[float] = probs_cpu.tolist()
        if isinstance(probs_list, float):
            probs_list = [probs_list]

        conf_thr = self.tier_confirmed_threshold
        susp_thr = self.tier_suspicious_threshold

        # Full probability vector — always present, no filtering.
        probabilities: dict[str, float] = {
            cls_name: round(prob, 4)
            for cls_name, prob in zip(self._class_names, probs_list)
        }

        # Tiered lists — sorted descending by probability within each tier.
        confirmed: list[dict] = []
        suspicious: list[dict] = []
        for cls_name, prob in zip(self._class_names, probs_list):
            p = round(prob, 4)
            if prob >= conf_thr:
                confirmed.append({"vulnerability_class": cls_name, "probability": p, "tier": "CONFIRMED"})
            elif prob >= susp_thr:
                suspicious.append({"vulnerability_class": cls_name, "probability": p, "tier": "SUSPICIOUS"})
        confirmed.sort(key=lambda x: x["probability"], reverse=True)
        suspicious.sort(key=lambda x: x["probability"], reverse=True)

        # Label reflects highest active tier.
        if confirmed:
            label = "confirmed_vulnerable"
        elif suspicious:
            label = "suspicious"
        else:
            label = "safe"

        # Legacy field: backward-compat alias for confirmed.
        # Old consumers reading result["vulnerabilities"] get CONFIRMED classes only,
        # which is the closest equivalent to the old above-per-class-threshold list.
        vulnerabilities = [
            {"vulnerability_class": v["vulnerability_class"], "probability": v["probability"]}
            for v in confirmed
        ]

        logger.info(
            f"Label: {label} | confirmed={len(confirmed)} suspicious={len(suspicious)} | "
            f"nodes={graph.num_nodes} edges={graph.num_edges} "
            f"truncated={tokens.get('truncated', False)} windows={windows_used}"
        )

        return {
            "label":            label,
            "probabilities":    probabilities,
            "confirmed":        confirmed,
            "suspicious":       suspicious,
            "vulnerabilities":  vulnerabilities,   # legacy
            "tier_thresholds":  {
                "confirmed":  conf_thr,
                "suspicious": susp_thr,
                "noteworthy": 0.10,
            },
            "thresholds":    self.thresholds.cpu().tolist(),
            "truncated":     tokens.get("truncated", False),
            "windows_used":  windows_used,
            "num_nodes":     int(graph.num_nodes),
            "num_edges":     int(graph.num_edges),
        }
