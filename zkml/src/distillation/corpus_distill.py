"""
corpus_distill.py — Train the proxy model directly from the 61-contract corpus.

Bypasses DualPathDataset (which requires pre-extracted .pt files in
ml/data/graphs/ and ml/data/tokens/ that the data module export didn't produce).
Instead, runs the full teacher predictor on each .sol contract to extract
128-dim CrossAttentionFusion embeddings and 10-class sigmoid probabilities,
then trains the proxy on that feature/target cache.

Usage:
    cd ~/projects/sentinel
    source ml/.venv/bin/activate
    python zkml/src/distillation/corpus_distill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.src.inference.predictor import Predictor
from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

CORPUS_ROOT       = "manual_hand_written_contracts"
PROXY_CHECKPOINT  = "zkml/models/proxy_best.pt"
TEACHER_CHECKPOINT = "ml/checkpoints/GCB-P1-Run12-v3dospatched-20260613_FINAL.pt"

BATCH_SIZE        = 64
EPOCHS            = 50
LR                = 1e-3
AGREEMENT_TARGET  = 0.95
THRESHOLD         = 0.50
RANDOM_SEED       = 42

# For a 61-contract corpus, use a generous val split
VAL_SPLIT = 0.25  # ~15 val, ~46 train

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def collect_contract_paths() -> list[Path]:
    """Recursively find all .sol files, excluding _quarantine/."""
    root = Path(CORPUS_ROOT)
    paths = []
    for sol_file in sorted(root.rglob("*.sol")):
        if "_quarantine" in str(sol_file):
            continue
        paths.append(sol_file)
    logger.info(f"Found {len(paths)} contracts in corpus (excl. quarantine)")
    return paths


def extract_features(predictor: Predictor, source_code: str, device: str):
    """
    Run teacher forward pass and capture fusion embedding + class logits.

    Uses model.forward(return_aux=True) which handles all internal
    mask reshaping, window-attention, and cross-attention. The fusion
    embedding is captured from aux_dict["fusion_embedding"].

    Returns:
        fusion:  [128] float tensor — CrossAttentionFusion output [B=1 squeezed]
        logits:  [10]  float tensor — raw classifier logits (pre-sigmoid) [squeezed]
    """
    from torch_geometric.data import Batch

    graph, windows = predictor.preprocessor.process_source_windowed(source_code)
    batch = Batch.from_data_list([graph]).to(device)

    n_real = len(windows)
    selected = windows[:4]
    pad_ids  = torch.zeros(1, 512, dtype=torch.long, device=device)
    pad_mask = torch.zeros(1, 512, dtype=torch.long, device=device)
    padded = list(selected)
    while len(padded) < 4:
        padded.append({"input_ids": pad_ids, "attention_mask": pad_mask})
    stacked_ids  = torch.cat(
        [w["input_ids"].to(device) for w in padded], dim=0
    ).unsqueeze(0)
    stacked_mask = torch.cat(
        [w["attention_mask"].to(device) for w in padded], dim=0
    ).unsqueeze(0)

    model = predictor.model
    model.eval()

    with torch.no_grad():
        output = model(batch, stacked_ids, stacked_mask, return_aux=True)
    logits, aux_dict = output
    fusion = aux_dict["fusion_embedding"]  # [1, 128]

    return fusion.squeeze(0).cpu(), logits.squeeze(0).cpu()


def compute_agreement(
    proxy_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    threshold: float = THRESHOLD,
) -> float:
    proxy_labels   = (proxy_scores   >= threshold).long()
    teacher_labels = (teacher_scores >= threshold).long()
    matches = (proxy_labels == teacher_labels).float()
    return matches.mean().item()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    torch.manual_seed(RANDOM_SEED)

    logger.info(f"Corpus distillation on: {device}")
    logger.info(f"Teacher: {TEACHER_CHECKPOINT}")
    logger.info(f"Circuit: {CIRCUIT_VERSION}")

    # ── Load teacher ──────────────────────────────────────────────────
    logger.info("Loading teacher...")
    predictor = Predictor(checkpoint=TEACHER_CHECKPOINT)
    logger.info(f"Teacher loaded — architecture: {predictor.architecture}")

    # ── Collect contracts ─────────────────────────────────────────────
    contract_paths = collect_contract_paths()
    if len(contract_paths) < 10:
        raise RuntimeError(
            f"Only {len(contract_paths)} contracts found — need at least 10 "
            f"for meaningful distillation. Corpus may have been moved."
        )

    # ── Extract features + targets from every contract ────────────────
    logger.info(f"Extracting teacher features from {len(contract_paths)} contracts...")
    fusion_list, score_list = [], []
    failed = 0

    for i, sol_path in enumerate(contract_paths):
        try:
            source = sol_path.read_text(encoding="utf-8", errors="replace")
            fusion, logits = extract_features(predictor, source, device)
            probs = torch.sigmoid(logits)  # [10]
            fusion_list.append(fusion)
            score_list.append(probs)
        except Exception as exc:
            failed += 1
            logger.warning(f"  Skipped {sol_path.name}: {exc}")
            continue

        if (i + 1) % 20 == 0:
            logger.info(f"  {i + 1}/{len(contract_paths)} contracts processed...")

    if failed > len(contract_paths) * 0.5:
        raise RuntimeError(
            f"{failed}/{len(contract_paths)} contracts failed extraction. "
            f"Teacher or preprocessor may be broken."
        )

    logger.info(
        f"Feature extraction complete — "
        f"{len(fusion_list)} successes, {failed} failures"
    )

    features = torch.stack(fusion_list)          # [N, 128]
    targets  = torch.stack(score_list)           # [N, 10]

    logger.info(
        f"Features shape: {features.shape} | "
        f"Target range: [{targets.min():.4f}, {targets.max():.4f}] | "
        f"Target mean: {targets.mean():.4f}"
    )

    # ── Train/val split ───────────────────────────────────────────────
    n = len(features)
    n_val = max(1, int(n * VAL_SPLIT))
    indices = torch.randperm(n)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_features = features[train_idx]
    train_targets  = targets[train_idx]
    val_features   = features[val_idx]
    val_targets    = targets[val_idx]

    logger.info(
        f"Split — train: {train_features.shape[0]}, val: {val_features.shape[0]}"
    )

    # ── Train proxy ───────────────────────────────────────────────────
    proxy     = ProxyModel().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(proxy.parameters(), lr=LR)

    best_agreement = 0.0
    Path("zkml/models").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        proxy.train()
        train_loss = 0.0

        for b in range(0, len(train_features), BATCH_SIZE):
            xb = train_features[b:b + BATCH_SIZE].to(device)
            yb = train_targets[b:b + BATCH_SIZE].to(device)

            optimiser.zero_grad()
            pred = proxy(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_features) // BATCH_SIZE)

        proxy.eval()
        with torch.no_grad():
            val_pred   = proxy(val_features.to(device)).cpu()
        agreement = compute_agreement(val_pred, val_targets)

        logger.info(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Agreement: {agreement:.4f} "
            f"({'TARGET MET' if agreement >= AGREEMENT_TARGET else f'target: {AGREEMENT_TARGET:.0%}'})"
        )

        if agreement > best_agreement:
            best_agreement = agreement
            torch.save(proxy.state_dict(), PROXY_CHECKPOINT)
            logger.info(f"  New best — agreement: {best_agreement:.4f}")

        if agreement >= AGREEMENT_TARGET:
            logger.info(f"Agreement target reached at epoch {epoch}.")
            break

    logger.info(f"Distillation complete — best agreement: {best_agreement:.4f}")
    logger.info(f"Checkpoint: {PROXY_CHECKPOINT}")
    logger.info(f"Circuit:    {CIRCUIT_VERSION}")

    if best_agreement < 0.90:
        logger.warning(
            f"Best agreement {best_agreement:.4f} below 0.90 minimum for ZK. "
            f"Causes: (1) 61-contract corpus may be too small for 128→64→32→10 "
            f"architecture, (2) teacher scores are concentrated near 0 or 1 "
            f"(easy agreement is inflated), (3) per-class distillation is "
            f"harder than scalar-mean distillation."
        )


if __name__ == "__main__":
    main()
