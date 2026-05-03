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
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import CLASS_NAMES

_BINARY_CLASS_NAME = "BinaryScore"

# ---------------------------------------------------------------------------
# Bug 4 fix — explicit architecture allowlist (replaces silent else-64 branch)
# ---------------------------------------------------------------------------
# Registry pattern: adding a new architecture = one dict entry, not an elif hunt.
# Keys must match exactly what trainer.py writes into checkpoint["config"]["architecture"].
_ARCH_TO_FUSION_DIM: dict[str, int] = {
    "cross_attention_lora": 128,
    "legacy":               64,
    "legacy_binary":        64,
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

    def __init__(
        self,
        checkpoint: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if not (0.0 < threshold < 1.0):
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}.")

        self.threshold = threshold  # kept for potential external use, not used in _score
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
            fusion_output_dim = _ARCH_TO_FUSION_DIM[architecture]

            # Strict metadata cross-check: fusion_output_dim
            cfg_dim = saved_cfg.get("fusion_output_dim")
            if cfg_dim is not None and cfg_dim != fusion_output_dim:
                raise ValueError(
                    f"Checkpoint config.fusion_output_dim={cfg_dim} does not match "
                    f"expected {fusion_output_dim} for architecture '{architecture}'. "
                    "Checkpoint may be corrupt or from an incompatible training run."
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

        # ------------------------------------------------------------------
        # Build model with correct architecture
        # ------------------------------------------------------------------
        self.model = SentinelModel(
            num_classes=num_classes,
            fusion_output_dim=fusion_output_dim,
            gnn_hidden_dim=saved_cfg.get("gnn_hidden_dim", 64),
            gnn_heads=saved_cfg.get("gnn_heads", 8),
            use_edge_attr=saved_cfg.get("use_edge_attr", True),
            gnn_edge_emb_dim=saved_cfg.get("gnn_edge_emb_dim", 16),
            lora_r=saved_cfg.get("lora_r", 8),
            lora_alpha=saved_cfg.get("lora_alpha", 16),
            lora_dropout=saved_cfg.get("lora_dropout", 0.1),
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
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
        Run one minimal forward pass with dummy tensors to surface CUDA
        and model-shape issues at startup instead of on the first real request.

        Audit fix #5 (2026-05-01) — 2-node 1-edge graph:
        The previous warmup used a single-node, zero-edge graph:
            edge_index = torch.zeros(2, 0, ...)
        A zero-edge graph never calls GATConv.propagate(), so attention
        coefficient shape bugs only appear on the first real contract.

        Fixed: 2 nodes, 1 undirected edge (0→1 and 1→0, so both directions
        are covered). This forces all three GATConv.propagate() calls to run
        and exercises the full message-passing code path at startup.

        Node feature dim (8) matches GNNEncoder's expected input_dim.
        Token tensor: first token real, rest PAD — avoids empty masked-mean.
        """
        try:
            # ── Audit fix #5: 2 nodes, 1 undirected edge (was 0 edges) ───────
            # Edge 0→1 and 1→0 = undirected. dim=8 matches GNNEncoder input_dim.
            dummy_x = torch.zeros(2, 8, dtype=torch.float32, device=self.device)
            dummy_edge_index = torch.tensor(
                [[0, 1], [1, 0]], dtype=torch.long, device=self.device
            )  # shape [2, 2] — two directed edges forming one undirected edge
            dummy_graph = Data(x=dummy_x, edge_index=dummy_edge_index)
            dummy_batch = Batch.from_data_list([dummy_graph]).to(self.device)

            # attention_mask: first token real, rest PAD — avoids empty masked mean
            dummy_ids = torch.zeros(1, 512, dtype=torch.long, device=self.device)
            dummy_mask = torch.zeros(1, 512, dtype=torch.long, device=self.device)
            dummy_mask[0, 0] = 1

            with torch.no_grad():
                _ = self.model(dummy_batch, dummy_ids, dummy_mask)

            logger.info("Warmup forward pass succeeded — model ready (2-node 1-edge graph)")
        except Exception as exc:
            raise RuntimeError(
                f"Model warmup failed — checkpoint may be incompatible with current code. "
                f"Error: {exc}"
            ) from exc

    def predict(self, sol_path: str | Path) -> dict:
        """Score a Solidity contract file on disk."""
        graph, tokens = self.preprocessor.process(sol_path)
        result = self._score(graph, tokens)
        result["windows_used"] = 1
        return result

    def predict_source(self, source_code: str, name: str = "contract") -> dict:
        """
        Score a raw Solidity source string.

        For contracts ≤ 512 tokens: single forward pass (same as before).
        For contracts > 512 tokens: sliding-window tokenization (T1-C).
            Each window is scored independently; class probabilities are aggregated
            by max across windows so late-file patterns (e.g. withdrawal logic at
            line 400+) are not silently truncated.
        """
        graph, windows = self.preprocessor.process_source_windowed(source_code)

        if len(windows) == 1:
            result = self._score(graph, windows[0])
            result["windows_used"] = 1
            return result

        return self._score_windowed(graph, windows)

    @staticmethod
    def _aggregate_window_predictions(
        probs_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate per-window probability tensors into a single class vector.

        Strategy: max probability per class across all windows.
        Rationale: a vulnerability is present if ANY window detects it above
        threshold. Taking the max preserves the strongest signal regardless of
        window position — a reentrancy buried at line 400 will not be diluted
        by averaging with the safe preamble at lines 1-100.
        """
        return torch.stack(probs_list).max(dim=0).values  # [num_classes]

    def _score_windowed(self, graph, windows: list[dict]) -> dict:
        """Run forward pass for each window, aggregate, and format result."""
        self.model.eval()
        batch = Batch.from_data_list([graph]).to(self.device)

        per_window_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for window in windows:
                input_ids = window["input_ids"].to(self.device)      # [1, 512]
                attention_mask = window["attention_mask"].to(self.device)  # [1, 512]
                logits = self.model(batch, input_ids, attention_mask)
                per_window_probs.append(torch.sigmoid(logits.float()).squeeze(0))

        agg_probs = self._aggregate_window_predictions(per_window_probs)  # [num_classes]

        # Use the truncated flag from the first window (it reflects the source length)
        first_window = windows[0]
        result = self._format_result(graph, agg_probs, first_window, len(windows))
        return result

    def _score(self, graph, tokens: dict) -> dict:
        """
        Run forward pass for a single token window and return structured result.

        Sigmoid applied here — NOT inside model (BCEWithLogitsLoss compatibility).
        Per‑class thresholds are applied instead of a single global threshold.

        Result schema (canonical — all consumers must read this shape):
            {
                "label": "vulnerable" | "safe",
                "vulnerabilities": [
                    {"vulnerability_class": str, "probability": float},
                    ...
                ],
                "threshold": float,   # fallback threshold (see note below)
                "truncated": bool,
                "windows_used": int,  # always 1 from this path
                "num_nodes": int,
                "num_edges": int,
            }

        Note on "threshold":
            When per-class thresholds are loaded from JSON, self.threshold is
            the fallback float and does NOT represent the actual decision
            boundaries used. Tracked as audit finding #6.
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
        """Convert probability tensor + metadata into the canonical result dict."""
        probs_cpu = probs.cpu()
        probs_list: list[float] = probs_cpu.tolist()
        if isinstance(probs_list, float):
            probs_list = [probs_list]

        # Bug 3 fix — emit canonical key 'vulnerability_class' (was 'class').
        vulnerabilities = [
            {"vulnerability_class": cls_name, "probability": round(prob, 4)}
            for cls_name, prob, thresh in zip(
                self._class_names, probs_list, self.thresholds.cpu()
            )
            if prob >= thresh.item()
        ]
        vulnerabilities.sort(key=lambda x: x["probability"], reverse=True)

        label = "vulnerable" if vulnerabilities else "safe"

        logger.info(
            f"Label: {label} | {len(vulnerabilities)} class(es) above thresholds | "
            f"nodes={graph.num_nodes} edges={graph.num_edges} "
            f"truncated={tokens.get('truncated', False)} windows={windows_used}"
        )

        return {
            "label": label,
            "vulnerabilities": vulnerabilities,
            "threshold": self.threshold,
            "truncated": tokens.get("truncated", False),
            "windows_used": windows_used,
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.num_edges),
        }
