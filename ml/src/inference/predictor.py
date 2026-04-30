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

FIXES (2026-04-29):
    Bug 4 — Unknown architecture now raises ValueError (was silent fallthrough to 64).
             _ARCH_TO_FUSION_DIM allowlist replaces if/else.
    Bug 5 — num_classes > len(CLASS_NAMES) now raises ValueError before slicing.
             Prevents zip() silent truncation if a future checkpoint adds new classes.
    Bug 3 — _score() emits 'vulnerability_class' key (was 'class').
             predictor.py is the canonical schema owner; no consumer remapping needed.

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
            # The old `else 64` silently loaded wrong weights when architecture
            # was misspelled or a new value was added — causing cryptic shape
            # mismatches deep in load_state_dict, or worse, silent wrong inference.
            if architecture not in _ARCH_TO_FUSION_DIM:
                raise ValueError(
                    f"Unknown checkpoint architecture: '{architecture}'. "
                    f"Known architectures: {list(_ARCH_TO_FUSION_DIM.keys())}. "
                    "Add the new architecture to _ARCH_TO_FUSION_DIM in predictor.py "
                    "and set the correct fusion_output_dim before loading this checkpoint."
                )
            fusion_output_dim = _ARCH_TO_FUSION_DIM[architecture]

            # Strict metadata cross-check: if checkpoint config explicitly records
            # fusion_output_dim, it must match the architecture's expected value.
            cfg_dim = saved_cfg.get("fusion_output_dim")
            if cfg_dim is not None and cfg_dim != fusion_output_dim:
                raise ValueError(
                    f"Checkpoint config.fusion_output_dim={cfg_dim} does not match "
                    f"expected {fusion_output_dim} for architecture '{architecture}'. "
                    "Checkpoint may be corrupt or from an incompatible training run."
                )

            # Strict metadata cross-check: if checkpoint records class_names, verify
            # they are a prefix-match of CLASS_NAMES (same order, no renames).
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
        # CLASS_NAMES[:11] returns 10 items without error when len(CLASS_NAMES)==10;
        # zip then stops at 10, silently dropping the 11th model output forever.
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

            # Build thresholds tensor aligned with self._class_names.
            # Warn per class when a threshold is absent — the fallback is used but
            # the operator should know which classes are relying on the default.
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
        # Warmup forward pass — catches CUDA / shape issues at startup,
        # not on the first real request. Uses minimal dummy tensors (no Slither).
        # ------------------------------------------------------------------
        self._warmup()
        logger.info(f"Predictor ready | {self.num_classes} classes | {architecture}")

    def _warmup(self) -> None:
        """
        Run one minimal forward pass with dummy tensors to surface CUDA
        and model-shape issues at startup instead of on the first real request.

        Uses a single-node graph and all-PAD token sequence — enough to exercise
        every layer path without requiring Slither or a real Solidity file.
        """
        try:
            dummy_x = torch.zeros(1, 8, dtype=torch.float32, device=self.device)
            dummy_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            dummy_graph = Data(x=dummy_x, edge_index=dummy_edge_index)
            dummy_batch = Batch.from_data_list([dummy_graph]).to(self.device)

            # attention_mask: first token real, rest PAD — avoids empty masked mean
            dummy_ids = torch.zeros(1, 512, dtype=torch.long, device=self.device)
            dummy_mask = torch.zeros(1, 512, dtype=torch.long, device=self.device)
            dummy_mask[0, 0] = 1

            with torch.no_grad():
                _ = self.model(dummy_batch, dummy_ids, dummy_mask)

            logger.info("Warmup forward pass succeeded — model ready")
        except Exception as exc:
            raise RuntimeError(
                f"Model warmup failed — checkpoint may be incompatible with current code. "
                f"Error: {exc}"
            ) from exc

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

        Result schema (canonical — all consumers must read this shape):
            {
                "label": "vulnerable" | "safe",
                "vulnerabilities": [
                    {"vulnerability_class": str, "probability": float},
                    ...
                ],
                "threshold": float,
                "truncated": bool,
                "num_nodes": int,
                "num_edges": int,
            }
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

        # Bug 3 fix — emit canonical key 'vulnerability_class' (was 'class').
        # predictor.py owns the schema; api.py, inference_server.py, nodes.py, and
        # any script calling predict_source() directly all read this key without remapping.
        vulnerabilities = [
            {"vulnerability_class": name, "probability": round(prob, 4)}
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
