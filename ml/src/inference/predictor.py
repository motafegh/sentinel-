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

    4. Per-class thresholds loaded from {checkpoint.stem}_thresholds.json
       Replaces single global threshold with class-specific decision boundaries.
       If file missing, falls back to user-supplied threshold (or 0.5).

    5. Stores self.architecture for health endpoint (avoid reloading checkpoint)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger
from torch_geometric.data import Batch

from ml.src.inference.preprocess import ContractPreprocessor
from ml.src.models.sentinel_model import SentinelModel
from ml.src.training.trainer import CLASS_NAMES

_BINARY_CLASS_NAME = "BinaryScore"


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

            # fusion_output_dim changed: cross_attention_lora=128, legacy=64
            fusion_output_dim = 128 if architecture == "cross_attention_lora" else 64

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
            logger.warning("Old-format checkpoint — loading as binary (num_classes=1)")

        self.num_classes = num_classes
        self.architecture = architecture          # stored for health endpoint
        self._class_names = CLASS_NAMES[:num_classes] if num_classes > 1 else [_BINARY_CLASS_NAME]

        # ------------------------------------------------------------------
        # Build model with correct architecture
        # ------------------------------------------------------------------
        self.model = SentinelModel(
            num_classes=num_classes,
            fusion_output_dim=fusion_output_dim,
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

            # Build thresholds tensor aligned with self._class_names
            per_class_thresholds = []
            for cls_name in self._class_names:
                thresh = class_thresholds_dict.get(cls_name, self.threshold)
                per_class_thresholds.append(thresh)

            self.thresholds = torch.tensor(
                per_class_thresholds, dtype=torch.float32, device=self.device
            )
            logger.info(
                f"Loaded per‑class thresholds from {thresholds_path} — "
                f"min={self.thresholds.min().item():.3f}, max={self.thresholds.max().item():.3f}"
            )
        else:
            self.thresholds = torch.full(
                (self.num_classes,), self.threshold, dtype=torch.float32, device=self.device
            )
            logger.warning(
                f"No thresholds JSON found at {thresholds_path} — "
                f"using uniform threshold {self.threshold} for all classes"
            )

        # ------------------------------------------------------------------
        # Preprocessor — loaded once, reused per call
        # ------------------------------------------------------------------
        self.preprocessor = ContractPreprocessor()
        logger.info(f"Predictor ready | {self.num_classes} classes | {architecture}")

    def predict(self, sol_path: str | Path) -> dict:
        """Score a Solidity contract file on disk."""
        graph, tokens = self.preprocessor.process(sol_path)
        return self._score(graph, tokens)

    def predict_source(self, source_code: str, name: str = "contract") -> dict:
        """Score a raw Solidity source string."""
        graph, tokens = self.preprocessor.process_source(source_code, name=name)
        return self._score(graph, tokens)

    def _score(self, graph, tokens: dict) -> dict:
        """
        Run forward pass, return structured multi-label result.

        Sigmoid applied here — NOT inside model (BCEWithLogitsLoss compatibility).
        Per‑class thresholds are applied instead of a single global threshold.
        """
        self.model.eval()

        with torch.no_grad():
            batch = Batch.from_data_list([graph]).to(self.device)
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)

            logits = self.model(batch, input_ids, attention_mask)  # [1, num_classes]
            probs = torch.sigmoid(logits)  # [1, num_classes]

        probs_list: list[float] = probs.squeeze(0).cpu().tolist()
        if isinstance(probs_list, float):
            probs_list = [probs_list]

        # Apply per‑class thresholds
        vulnerabilities = [
            {"class": name, "probability": round(prob, 4)}
            for name, prob, thresh in zip(self._class_names, probs_list, self.thresholds.cpu())
            if prob >= thresh.item()
        ]
        vulnerabilities.sort(key=lambda x: x["probability"], reverse=True)

        label = "vulnerable" if vulnerabilities else "safe"

        logger.info(
            f"Label: {label} | {len(vulnerabilities)} class(es) above thresholds | "
            f"nodes={graph.num_nodes} edges={graph.num_edges} "
            f"truncated={tokens['truncated']}"
        )

        return {
            "label": label,
            "vulnerabilities": vulnerabilities,
            "threshold": self.threshold,
            "truncated": tokens["truncated"],
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.num_edges),
        }