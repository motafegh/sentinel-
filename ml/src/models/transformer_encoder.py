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
       - Defaults updated from original: r=16, alpha=32, dropout=0.1, ["query","value"]

WHY LoRA:
    Full fine-tune: 125M params → OOM on 8GB VRAM, catastrophic forgetting on 68K contracts
    Frozen:         0 trainable → CodeBERT never adapts to vulnerability semantics
    LoRA:           590K trainable (r=16) → adapts query+value attention to security patterns
                    without touching the 125M frozen weights

PARAMETER COUNT (current config r=16, alpha=32):
    Frozen (CodeBERT backbone):  124,705,536  (unchanged, never updated)
    Trainable (LoRA matrices):       589,824  (~590K across 12 layers × Q+V at r=16)
    Scale factor (alpha/r):              2.0  (lora_alpha=32, r=16)

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

        # Load pretrained CodeBERT — 125M parameters.
        # flash_attention_2 uses tiled CUDA kernels that avoid materialising the
        # full [B*W,512,512] attention matrix. Falls back to sdpa if unavailable.
        # Must be set before get_peft_model so LoRA sees the correct implementation.
        # Save and restore the global default dtype around BERT loading.
        # from_pretrained with torch_dtype=bfloat16 calls torch.set_default_dtype
        # as a side effect, which pollutes any nn.Linear created afterwards with BF16 weights.
        _prev_default_dtype = torch.get_default_dtype()
        try:
            self.bert = AutoModel.from_pretrained(
                "microsoft/graphcodebert-base",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            logger.info("TransformerEncoder — Flash Attention 2 active")
        except ImportError:
            # [A28] Narrow to ImportError only. ValueError means a corrupted config.json or
            # missing model file — that is a real error and must propagate, not fall back silently.
            self.bert = AutoModel.from_pretrained(
                "microsoft/graphcodebert-base",
                attn_implementation="sdpa",
            )
            logger.info("TransformerEncoder — SDPA active (flash-attn unavailable)")
        finally:
            torch.set_default_dtype(_prev_default_dtype)

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

        # [A30] Validate _word_embeddings path at construction time — surfaces PEFT layout
        # changes immediately rather than crashing at the first forward pass using prefix injection.
        try:
            _ = self._word_embeddings
        except AttributeError as _we_err:
            raise RuntimeError(
                f"TransformerEncoder: PEFT word embedding path validation failed. {_we_err}"
            ) from _we_err

    @property
    def _word_embeddings(self) -> nn.Embedding:
        """
        Word embedding layer of the underlying GraphCodeBERT model.

        [A30] Tries multiple known PEFT internal paths in precedence order so PEFT version
        changes do not silently return a wrong object or crash mid-forward. Validated at
        __init__ time — failures surface at construction, not at the first forward pass.
        """
        _paths = [
            "base_model.model.embeddings.word_embeddings",   # PEFT ≥0.4 (LoraModel.model)
            "model.embeddings.word_embeddings",               # some PEFT variants
            "base_model.embeddings.word_embeddings",          # older PEFT ≤0.3
        ]
        for path in _paths:
            obj = self.bert
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Embedding):
                    return obj
            except AttributeError:
                continue
        raise AttributeError(
            f"[A30] _word_embeddings: could not locate word embedding nn.Embedding "
            f"via any of {_paths}. PEFT version may have changed the internal layout. "
            "Update candidate paths in transformer_encoder.py."
        )

    def forward(
        self,
        input_ids:          torch.Tensor,                   # [B, L] or [B, W, L]
        attention_mask:     torch.Tensor,                   # [B, L] or [B, W, L]
        gnn_prefix_nodes:   Optional[torch.Tensor] = None,  # [B, K, 768] or None
        gnn_prefix_counts:  Optional[torch.Tensor] = None,  # [B] real node counts, IMP-M3
        output_attentions:  bool = False,                   # IMP-M2: return prefix_attn_mean too
    ):
        """
        Run GraphCodeBERT + LoRA forward pass and return all token embeddings.

        Accepts both single-window [B, L] and multi-window [B, W, L] inputs.
        When gnn_prefix_nodes is provided ([B, K, 768]), injects K GNN-derived
        prefix tokens before the code tokens using inputs_embeds.  The total
        sequence length stays L (K prefix + L-K code).  CLS moves to position K.

        Multi-window: windows are flattened into the batch dim, passed through
        GraphCodeBERT in one fused call (B*W sequences), then reassembled.
        Prefix is shared across all windows (same K nodes per contract).

        Position IDs with prefix:
            Prefix tokens:  position_id=1  (RoBERTa padding pos — no positional bias)
            Code tokens:    position_ids 3..3+(L-K-1)  (CLS at 3, then 4, 5, ...)
            Max position:   3+(L-K-1)  well within RoBERTa's 514 limit for K≥2.

        Gradient flow:
            Frozen weights (requires_grad=False): PyTorch skips backward nodes.
            LoRA A/B matrices (requires_grad=True): gradients flow normally.
            peft manages this split internally; no manual no_grad() is needed.

        Returns (output_attentions=False, default):
            [B, L, 768] or [B, W*L, 768] — all token embeddings

        Returns (output_attentions=True, prefix path only):
            (last_hidden_state, prefix_attn_mean: float)
            prefix_attn_mean = mean attention weight code tokens → prefix positions,
            averaged over all layers, heads, and sequences. Near zero (< 0.002 for
            5+ epochs) means the transformer is ignoring the prefix (IMP-M2 gate).
        """
        if gnn_prefix_nodes is None:
            # Standard path — no prefix overhead
            if input_ids.dim() == 2:
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state  # [B, L, 768]

            # Multi-window: [B, W, L] → [B*W, L] → GraphCodeBERT → [B, W*L, 768]
            B, W, L = input_ids.shape
            flat_ids  = input_ids.view(B * W, L)
            flat_mask = attention_mask.view(B * W, L)
            outputs   = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
            return outputs.last_hidden_state.view(B, W * L, 768)

        # ── Prefix injection path ─────────────────────────────────────────────
        K = gnn_prefix_nodes.shape[1]

        if input_ids.dim() == 2:
            B, L        = input_ids.shape
            code_budget = L - K

            code_ids  = input_ids[:, :code_budget]       # [B, L-K] keep CLS at position 0
            code_mask = attention_mask[:, :code_budget]  # [B, L-K]

            word_embs = self._word_embeddings(code_ids).to(dtype=gnn_prefix_nodes.dtype)
            inputs_embeds = torch.cat([gnn_prefix_nodes, word_embs], dim=1)  # [B, L, 768]

            # IMP-M3: use actual node counts so zero-padded prefix positions are masked.
            # 95.5% of contracts fill all K slots (count==K) — this is a no-op for them.
            if gnn_prefix_counts is not None:
                # [A29] Vectorised: broadcast arange [K] vs counts [B] — no Python loop over B.
                _arange = torch.arange(K, device=attention_mask.device)
                prefix_mask = (_arange.unsqueeze(0) < gnn_prefix_counts.unsqueeze(1)).to(attention_mask.dtype)
            else:
                prefix_mask = torch.ones(B, K, dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask    = torch.cat([prefix_mask, code_mask], dim=1)        # [B, L]

            prefix_pos   = input_ids.new_ones(B, K)                          # pos_id=1 (pad slot)
            code_pos     = torch.arange(3, 3 + code_budget, dtype=torch.long,
                                        device=input_ids.device).unsqueeze(0).expand(B, -1)
            position_ids = torch.cat([prefix_pos, code_pos], dim=1)          # [B, L]

            outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            if output_attentions and outputs.attentions is not None:
                # attentions: tuple of 12 tensors, each [B, heads, L, L]
                # Slice code→prefix: rows K:L (code positions) × cols :K (prefix positions)
                attn = torch.stack(list(outputs.attentions), dim=0)  # [12, B, heads, L, L]
                prefix_attn_mean = attn[:, :, :, K:, :K].mean().item()
                return outputs.last_hidden_state, prefix_attn_mean
            return outputs.last_hidden_state  # [B, L, 768]

        # Multi-window with prefix: [B, W, L] — shared prefix across all windows
        B, W, L     = input_ids.shape
        code_budget = L - K

        flat_ids  = input_ids[:, :, :code_budget].reshape(B * W, code_budget)   # [B*W, L-K]
        flat_mask = attention_mask[:, :, :code_budget].reshape(B * W, code_budget)

        word_embs = self._word_embeddings(flat_ids).to(dtype=gnn_prefix_nodes.dtype)  # [B*W, L-K, 768]

        # Expand prefix: [B, K, 768] → [B*W, K, 768]
        prefix_expanded = (
            gnn_prefix_nodes.unsqueeze(1).expand(-1, W, -1, -1).reshape(B * W, K, 768)
        )
        inputs_embeds = torch.cat([prefix_expanded, word_embs], dim=1)  # [B*W, L, 768]

        # IMP-M3: mask zero-padded prefix positions per-graph, expanded across windows
        if gnn_prefix_counts is not None:
            # [A29] Vectorised: broadcast arange [K] vs counts [B] — no Python loop over B.
            _arange = torch.arange(K, device=flat_mask.device)
            prefix_mask = (_arange.unsqueeze(0) < gnn_prefix_counts.unsqueeze(1)).to(flat_mask.dtype)
            prefix_mask = prefix_mask.unsqueeze(1).expand(-1, W, -1).reshape(B * W, K)
        else:
            prefix_mask = torch.ones(B * W, K, dtype=flat_mask.dtype, device=flat_mask.device)
        full_mask    = torch.cat([prefix_mask, flat_mask], dim=1)        # [B*W, L]

        prefix_pos   = flat_ids.new_ones(B * W, K)
        code_pos     = torch.arange(3, 3 + code_budget, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0).expand(B * W, -1)
        position_ids = torch.cat([prefix_pos, code_pos], dim=1)          # [B*W, L]

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        if output_attentions and outputs.attentions is not None:
            attn = torch.stack(list(outputs.attentions), dim=0)  # [12, B*W, heads, L, L]
            prefix_attn_mean = attn[:, :, :, K:, :K].mean().item()
            return outputs.last_hidden_state.view(B, W * L, 768), prefix_attn_mean
        return outputs.last_hidden_state.view(B, W * L, 768)


class WindowAttentionPooler(nn.Module):
    """
    Pool W window-CLS embeddings into a single vector via learned attention.

    In multi-window mode TransformerEncoder returns [B, W*L, 768].  The CLS token
    of window i is at position i*window_size + prefix_k.  This module extracts
    those W CLS vectors and produces a weighted sum using a learned score function.

    Args:
        hidden_dim:   Embedding width (default 768 for GraphCodeBERT).
        window_size:  Tokens per window (default 512 = MAX_TOKEN_LENGTH).
        prefix_k:     Number of GNN prefix tokens prepended per window (default 0).
                      When prefix injection is active, CLS of window i is at
                      i*window_size + prefix_k instead of i*window_size.

    Single-window fallback: if W*L == window_size, returns CLS directly — zero
    overhead, no learned weights invoked.
    """

    def __init__(self, hidden_dim: int = 768, window_size: int = 512, prefix_k: int = 0) -> None:
        super().__init__()
        self.window_size = window_size
        self.prefix_k    = prefix_k
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, token_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embs: [B, W*L, 768] or [B, L, 768]
        Returns:
            [B, 768] — attention-weighted window-CLS pooling
        """
        B, WL, D = token_embs.shape
        if WL <= self.window_size:
            return token_embs[:, self.prefix_k, :]  # single-window: CLS at prefix_k
        W = WL // self.window_size
        # CLS of window i is at i*window_size + prefix_k
        cls_indices = torch.arange(W, device=token_embs.device) * self.window_size + self.prefix_k
        window_cls = token_embs[:, cls_indices, :]   # [B, W, 768]
        scores  = self.attn(window_cls)              # [B, W, 1]
        weights = torch.softmax(scores, dim=1)       # [B, W, 1]
        return (weights * window_cls).sum(dim=1)     # [B, 768]
