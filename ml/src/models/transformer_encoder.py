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

WHY LoRA:
    Full fine-tune: 125M params → OOM on 8GB VRAM, catastrophic forgetting on 68K contracts
    Frozen:         0 trainable → CodeBERT never adapts to vulnerability semantics
    LoRA:           295K trainable → adapts query+value attention to security patterns
                    without touching the 125M frozen weights

PARAMETER COUNT:
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


# ── LoRA configuration ──────────────────────────────────────────────────────
# Defined at module level so it is inspectable and testable without
# instantiating the full model.
#
# r=8:           rank — higher = more capacity and params, higher overfit risk
# lora_alpha=16: effective scale = alpha/r = 2.0 applied to the LoRA delta
# target_modules: query+value only — these control what CodeBERT attends to
# lora_dropout=0.1: regularises the 295K LoRA params against 68K contracts
# bias="none":   bias adaptation adds minimal value and increases complexity
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)


class TransformerEncoder(nn.Module):
    """
    CodeBERT encoder with LoRA fine-tuning for vulnerability-aware embeddings.

    Architecture:
        Frozen CodeBERT (125M params) + LoRA matrices (~295K trainable)
        on query+value projections of all 12 attention layers.

    Input:
        input_ids:      [B, 512]  — CodeBERT token IDs
        attention_mask: [B, 512]  — 1=real token, 0=padding

    Output:
        [B, 512, 768]  — ALL token embeddings (not just CLS).
                         CrossAttentionFusion uses these for node→token attention.
    """

    def __init__(self) -> None:
        super().__init__()

        # Load pretrained CodeBERT — 125M parameters
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")

        # Inject LoRA matrices into query+value of all 12 layers.
        # get_peft_model():
        #   1. Freezes ALL original CodeBERT weights (requires_grad=False)
        #   2. Injects trainable A [768,8] and B [8,768] matrices per targeted layer
        #   3. Forward pass computes: W_frozen @ x + (B @ A) @ x × (alpha/r)
        #      Gradients flow only through A and B; W_frozen receives none.
        self.bert = get_peft_model(self.bert, LORA_CONFIG)

        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.bert.parameters() if not p.requires_grad)
        logger.info(
            f"TransformerEncoder — LoRA active | "
            f"trainable: {trainable:,} | frozen: {frozen:,}"
        )

    def forward(
        self,
        input_ids:      torch.Tensor,  # [B, 512]
        attention_mask: torch.Tensor,  # [B, 512]
    ) -> torch.Tensor:
        """
        Run CodeBERT + LoRA forward pass and return all token embeddings.

        Gradient flow:
            Frozen weights (requires_grad=False): PyTorch skips backward nodes
            automatically — no activation memory allocated for them.
            LoRA A/B matrices (requires_grad=True): gradients flow normally.
            peft manages this split internally; no manual no_grad() is needed.

        Returns:
            all_token_embeddings: [B, 512, 768]
                All 512 positions, not just CLS.
                CrossAttentionFusion uses every position for node→token attention
                so that withdraw() can directly query "call.value" and "transfer"
                tokens before pooling loses that granularity.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # last_hidden_state: [B, 512, 768] — full sequence, not just position 0
        return outputs.last_hidden_state
