"""
transformer_encoder.py — Transformer Encoder for SENTINEL (Cross-Attention Upgrade)

WHAT CHANGED FROM ORIGINAL:
    1. LoRA fine-tuning added (peft library)
       - All 125M CodeBERT weights remain frozen
       - ~295K trainable LoRA matrices injected into query+value of all 12 layers
       - CodeBERT now adapts to vulnerability semantics without catastrophic forgetting
       - Requires: pip install peft  ← hard requirement, missing peft raises RuntimeError

    2. Returns ALL token embeddings instead of CLS token only
       - BEFORE: outputs.last_hidden_state[:, 0, :]  → [B, 768]
       - AFTER:  outputs.last_hidden_state            → [B, 512, 768]
       - WHY: CrossAttentionFusion needs all 512 token embeddings so each
         GNN node can query which tokens are most relevant to it.
         CLS is a blurry summary — withdraw() needs to find "call.value"
         and "transfer" specifically, not an averaged contract embedding.

    3. LoRA hyperparameters now passed as constructor arguments (P0-A refactor)
       - Removed module-level LORA_CONFIG constant
       - r, lora_alpha, lora_dropout, target_modules are configurable via TrainConfig
       - Defaults unchanged from original: r=8, alpha=16, dropout=0.1, ["query","value"]

WHY LoRA:
    Full fine-tune: 125M params → OOM on 8GB VRAM, catastrophic forgetting on 68K contracts
    Frozen:         0 trainable → CodeBERT never adapts to vulnerability semantics
    LoRA:           295K trainable → adapts query+value attention to security patterns
                    without touching the 125M frozen weights

PARAMETER COUNT (defaults):
    Frozen (CodeBERT backbone):  124,705,536  (unchanged, never updated)
    Trainable (LoRA matrices):       295,296  (~295K across 12 layers × Q+V)
    Scale factor (alpha/r):              2.0  (lora_alpha=16, r=8)

NOTE — why there is no torch.no_grad() around self.bert():
    peft's get_peft_model() marks every original CodeBERT weight with
    requires_grad=False. PyTorch's autograd engine does NOT build backward
    graph nodes for ops whose inputs are all requires_grad=False, so those
    frozen paths consume no gradient or activation memory.
    Wrapping the ENTIRE self.bert() call in no_grad() would also cut gradient
    flow to the LoRA A/B matrices that live inside the same forward pass —
    silently killing LoRA training. The gradient split is handled correctly
    by peft internally; no manual no_grad() scope is needed or safe here.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ── Hard requirement check ──────────────────────────────────────────────────
# Reviewed item #8: a silent warning-then-fallback means you can train 68K
# contracts with 0 trainable transformer parameters and only discover it at
# evaluation time. Raise immediately so the problem is impossible to miss.
if not _PEFT_AVAILABLE:
    raise RuntimeError(
        "peft library is required for TransformerEncoder but is not installed.\n"
        "Install it with:  pip install peft\n"
        "LoRA is not optional — without it CodeBERT has 0 trainable parameters "
        "and cannot adapt to vulnerability semantics."
    )


class TransformerEncoder(nn.Module):
    """
    CodeBERT encoder with LoRA fine-tuning for vulnerability-aware embeddings.

    Architecture:
        Frozen CodeBERT (125M params) + LoRA matrices (~590K trainable at r=16)
        on query+value projections of all 12 attention layers.

    Args:
        lora_r:              LoRA rank. Default 16.
        lora_alpha:          LoRA scale factor. Default 32.
        lora_dropout:        Dropout on LoRA paths. Default 0.1.
        lora_target_modules: Which attention projections to adapt.

    Input (single-window legacy):
        input_ids:      [B, L]     — token IDs  (L=512)
        attention_mask: [B, L]     — 1=real, 0=pad

    Input (multi-window):
        input_ids:      [B, W, L]  — W windows of L=512 tokens each
        attention_mask: [B, W, L]  — 1=real, 0=pad

    Output:
        Single-window: [B, L, 768]      — all token embeddings
        Multi-window:  [B, W×L, 768]    — all windows concatenated along seq dim
                       First L positions are window 0 (index 0 = CLS of first window).
    """

    def __init__(
        self,
        lora_r:              int       = 16,
        lora_alpha:          int       = 32,
        lora_dropout:        float     = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["query", "value"]
        elif isinstance(lora_target_modules, str):
            # Guard: MLflow may deserialise list as comma-joined string
            lora_target_modules = [s.strip() for s in lora_target_modules.split(",")]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        # Load pretrained CodeBERT — 125M parameters
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")

        # Inject LoRA matrices into targeted attention projections.
        # get_peft_model():
        #   1. Freezes ALL original CodeBERT weights (requires_grad=False)
        #   2. Injects trainable A [768,r] and B [r,768] matrices per targeted layer
        #   3. Forward pass computes: W_frozen @ x + (B @ A) @ x × (alpha/r)
        #      Gradients flow only through A and B; W_frozen receives none.
        self.bert = get_peft_model(self.bert, lora_config)

        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.bert.parameters() if not p.requires_grad)
        logger.info(
            f"TransformerEncoder — LoRA active | r={lora_r} alpha={lora_alpha} "
            f"modules={lora_target_modules} | "
            f"trainable: {trainable:,} | frozen: {frozen:,}"
        )

    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, L] or [B, W, L]
        attention_mask: torch.Tensor,  # [B, L] or [B, W, L]
    ) -> torch.Tensor:
        """
        Run CodeBERT + LoRA forward pass and return all token embeddings.

        Accepts both single-window [B, L] and multi-window [B, W, L] inputs.
        Multi-window: windows are flattened into the batch dim, passed through
        CodeBERT in one fused call (B*W sequences), then reassembled so each
        batch item has W×L token positions.  Window 0 position 0 is the CLS
        token for the first 512 tokens of the contract.

        Gradient flow:
            Frozen weights (requires_grad=False): PyTorch skips backward nodes.
            LoRA A/B matrices (requires_grad=True): gradients flow normally.
            peft manages this split internally; no manual no_grad() is needed.

        Returns:
            Single-window: [B, L, 768]    — all L positions, CLS at [:, 0, :]
            Multi-window:  [B, W*L, 768]  — windows concatenated along seq dim,
                                            CLS of window 0 at [:, 0, :]
        """
        if input_ids.dim() == 2:
            # Legacy single-window path — no reshape overhead
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state  # [B, L, 768]

        # Multi-window path: [B, W, L] → [B*W, L] → CodeBERT → [B, W*L, 768]
        B, W, L = input_ids.shape
        flat_ids  = input_ids.view(B * W, L)
        flat_mask = attention_mask.view(B * W, L)
        outputs   = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        # Reassemble: [B*W, L, 768] → [B, W*L, 768]
        return outputs.last_hidden_state.view(B, W * L, 768)
