"""
train_proxy.py — Knowledge Distillation Training Loop

Trains the ProxyModel (student) to mimic the full SentinelModel (teacher).

RECALL — what knowledge distillation means here:
    The teacher (full SentinelModel, ~125M params) already knows how to
    score contracts. We don't retrain it. We run it over all training data,
    collect its outputs, and train the tiny proxy to reproduce those outputs.
    The proxy never sees ground truth labels — it learns the teacher's
    behaviour. This is the "distillation" — the teacher's knowledge gets
    compressed into a much smaller form.

RECALL — why we use CrossAttentionFusion output as proxy input, not raw Solidity:
    The teacher's CrossAttentionFusion output [B, 128] already encodes everything:
    GNN structural understanding + CodeBERT semantic understanding, fused via
    bidirectional cross-attention. The proxy receives the teacher's complete
    understanding in 128 numbers. Mapping 128 rich features to a scalar is easy —
    that's why agreement converges rapidly.

RECALL — why labels are discarded (the _ in the loop):
    This is intentional distillation behaviour, not a bug.
    We are not training against ground truth labels.
    We are training the proxy to match the TEACHER'S scores.
    The teacher's score IS the training signal. Ground truth is irrelevant here.
    The teacher already learned from ground truth during Module 1 training.

RECALL — why MSE loss, not BCE or FocalLoss:
    BCE/FocalLoss compare predictions against binary labels (0 or 1).
    Here we're comparing proxy_score against teacher_score — two floats.
    MSE measures distance between two continuous values. That's the right
    loss for "make the proxy output match the teacher output numerically."

RECALL — multi-label distillation target:
    The teacher produces 10 logits (one per vulnerability class).
    We apply sigmoid to get per-class probabilities [B, 10], then take
    the mean across classes → [B] scalar per contract.
    This single scalar is the distillation target: it captures the teacher's
    overall confidence level and maps cleanly to the proxy's [B] output.

Usage:
    cd ~/projects/sentinel
    poetry run python zkml/src/distillation/train_proxy.py

Output:
    zkml/models/proxy_best.pt  ← saved when agreement ≥ 95%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ml.src.datasets.dual_path_dataset import DualPathDataset, dual_path_collate_fn
from ml.src.models.sentinel_model import SentinelModel
from zkml.src.distillation.proxy_model import CIRCUIT_VERSION, ProxyModel

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

TEACHER_CHECKPOINT = "ml/checkpoints/multilabel_crossattn_best.pt"
PROXY_CHECKPOINT   = "zkml/models/proxy_best.pt"

GRAPHS_DIR  = "ml/data/graphs"
TOKENS_DIR  = "ml/data/tokens"
SPLITS_DIR  = "ml/data/splits"

BATCH_SIZE       = 64
EPOCHS           = 50
LR               = 1e-3
AGREEMENT_TARGET = 0.95
THRESHOLD        = 0.50

# RECALL — reproducibility:
#   Without a fixed seed, two runs on the same data produce
#   different weight initialisations → different final checkpoints.
#   That makes it impossible to compare runs or debug regressions.
#   A fixed seed guarantees: same data + same seed = same result.
RANDOM_SEED = 42


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    teacher: SentinelModel,
    graphs,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract 128-dim CrossAttentionFusion outputs and distillation targets from the teacher.

    RECALL — why @torch.no_grad():
        Teacher weights are frozen — we never update them.
        No gradient computation needed → faster + less GPU memory.
        Gradients are only needed during proxy training, not here.

    RECALL — why we intercept at CrossAttentionFusion, not at the classifier output:
        teacher.classifier output = 10 logits per contract  [B, 10]
        teacher.fusion output     = 128-dim rich representation [B, 128]

        The proxy trains on 128 features, not 10. More information →
        easier learning task → higher agreement rate.

    RECALL — multi-label distillation target (ADR-025, Track 3):
        teacher.classifier gives [B, 10] raw logits.
        sigmoid([B, 10]) → per-class probabilities.
        mean(dim=1) → [B] scalar: average confidence across all 10 classes.
        This single scalar is the proxy's training target.

    Returns:
        features: [B, 128] — CrossAttentionFusion outputs (proxy inputs)
        scores:   [B]      — teacher's mean sigmoid score (proxy distillation target)
    """
    teacher.eval()

    graphs         = graphs.to(device)
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # GNN path: returns (node_embs [N, 64], batch [N])
    node_embs, batch = teacher.gnn(graphs.x, graphs.edge_index, graphs.batch)

    # Transformer path: returns [B, 512, 768]
    transformer_out = teacher.transformer(input_ids, attention_mask)

    # CrossAttentionFusion: needs node_embs, batch, token_embs, attention_mask
    # Returns [B, 128]
    features = teacher.fusion(node_embs, batch, transformer_out, attention_mask)

    # Multi-label teacher score: sigmoid([B, 10]).mean(dim=1) → [B]
    # This is the proxy's training target — teacher's mean confidence.
    scores = torch.sigmoid(teacher.classifier(features)).mean(dim=1)  # [B]

    return features.cpu(), scores.cpu()


# ------------------------------------------------------------------
# Agreement rate
# ------------------------------------------------------------------

def compute_agreement(
    proxy_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    threshold: float = THRESHOLD,
) -> float:
    """
    Fraction of contracts where proxy and teacher produce the same label.

    RECALL — why this is the real metric, not MSE:
        MSE going down means the proxy scores are numerically closer.
        But what we care about for the ZK system is: do proxy and teacher
        agree on the binary decision — vulnerable or safe?
        Agreement rate = fraction of contracts where both labels match.
        Target ≥95% means at most 5% of audits get a different label.
    """
    proxy_labels   = (proxy_scores   >= threshold).long()
    teacher_labels = (teacher_scores >= threshold).long()
    return (proxy_labels == teacher_labels).float().mean().item()


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Full knowledge distillation training loop.

    Steps:
        1. Fix random seed for reproducibility
        2. Load teacher (frozen)
        3. Load data via DualPathDataset (same as Module 1 trainer)
        4. Extract teacher features once — cache for all epochs
        5. Train proxy with MSE loss against teacher scores
        6. Evaluate agreement each epoch
        7. Save best checkpoint, stop early if target met
    """
    # ------------------------------------------------------------------
    # Step 1 — Fix seed
    # ------------------------------------------------------------------
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    logger.info(f"Random seed fixed: {RANDOM_SEED} — run is reproducible")
    logger.info(f"Starting proxy distillation on: {device}")
    logger.info(f"Teacher checkpoint: {TEACHER_CHECKPOINT}")
    logger.info(f"Agreement target:   {AGREEMENT_TARGET:.0%}")
    logger.info(f"Circuit version:    {CIRCUIT_VERSION}")

    # ------------------------------------------------------------------
    # Step 2 — Load teacher (frozen)
    # ------------------------------------------------------------------
    # Checkpoint format: {"model": state_dict, "config": {...}, ...}
    # The "config" dict carries architecture params (fusion_output_dim, num_classes).
    checkpoint = torch.load(
        TEACHER_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        config     = checkpoint.get("config", {})
    else:
        state_dict = checkpoint   # legacy: bare state_dict
        config     = {}

    num_classes       = config.get("num_classes", 10)
    fusion_output_dim = config.get("fusion_output_dim", 128)

    teacher = SentinelModel(
        num_classes=num_classes,
        fusion_output_dim=fusion_output_dim,
    ).to(device)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    logger.info(
        f"Teacher loaded and frozen — "
        f"num_classes={num_classes} fusion_output_dim={fusion_output_dim}"
    )

    # ------------------------------------------------------------------
    # Step 3 — Load data via DualPathDataset
    # ------------------------------------------------------------------
    # RECALL — DualPathDataset loads per-contract .pt files by index.
    # Same data source as Module 1 trainer — same contracts, same splits.
    # dual_path_collate_fn is required: without it, variable-size PyG
    # graphs crash the default collate trying to torch.stack different sizes.
    train_indices = np.load(f"{SPLITS_DIR}/train_indices.npy")
    val_indices   = np.load(f"{SPLITS_DIR}/val_indices.npy")

    train_dataset = DualPathDataset(
        graphs_dir=GRAPHS_DIR,
        tokens_dir=TOKENS_DIR,
        indices=train_indices.tolist(),
    )
    val_dataset = DualPathDataset(
        graphs_dir=GRAPHS_DIR,
        tokens_dir=TOKENS_DIR,
        indices=val_indices.tolist(),
    )

    # Using distinct names to avoid confusion with the TensorDataset
    # loaders created after feature extraction below.
    # These loaders feed the teacher for feature extraction only.
    teacher_train_loader = PyGDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,           # no shuffle — we cache features by position
        collate_fn=dual_path_collate_fn,
    )
    teacher_val_loader = PyGDataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dual_path_collate_fn,
    )

    logger.info(
        f"Dataset loaded — "
        f"train: {len(train_dataset)} contracts, "
        f"val: {len(val_dataset)} contracts"
    )

    # ------------------------------------------------------------------
    # Step 4 — Extract teacher features (one-time)
    # ------------------------------------------------------------------
    # RECALL — why we cache instead of re-running every epoch:
    #   The teacher has 124M frozen parameters.
    #   Running it every epoch = 50 × expensive forward passes.
    #   Teacher weights never change → features never change.
    #   Compute once, build TensorDataset, train proxy from cache.
    #   This is what makes 50 epochs practical on a single GPU.
    #
    # RECALL — labels are discarded here (the _) intentionally.
    #   Ground truth labels are irrelevant for distillation.
    #   The teacher's score IS the training target.
    logger.info("Extracting teacher features (one-time)...")

    train_features_list, train_scores_list = [], []
    val_features_list,   val_scores_list   = [], []

    for graphs, tokens, _ in teacher_train_loader:
        # _ = ground truth labels — intentionally discarded.
        # We train the proxy against teacher scores, not ground truth.
        feats, scores = extract_features(
            teacher,
            graphs,
            tokens["input_ids"],
            tokens["attention_mask"],
            device,
        )
        train_features_list.append(feats)
        train_scores_list.append(scores)

    for graphs, tokens, _ in teacher_val_loader:
        feats, scores = extract_features(
            teacher,
            graphs,
            tokens["input_ids"],
            tokens["attention_mask"],
            device,
        )
        val_features_list.append(feats)
        val_scores_list.append(scores)

    train_features       = torch.cat(train_features_list)   # [N_train, 128]
    train_teacher_scores = torch.cat(train_scores_list)     # [N_train]
    val_features         = torch.cat(val_features_list)     # [N_val, 128]
    val_teacher_scores   = torch.cat(val_scores_list)       # [N_val]

    logger.info(
        f"Features extracted — "
        f"train: {train_features.shape}, "
        f"val: {val_features.shape}"
    )

    # ------------------------------------------------------------------
    # Step 5 — Build proxy DataLoaders from cached features
    # ------------------------------------------------------------------
    # TensorDataset is much simpler than DualPathDataset —
    # just pairs of (128-dim features, teacher distillation score).
    # No graphs, no tokens, no collate_fn needed.
    proxy_train_dataset = TensorDataset(train_features, train_teacher_scores)
    proxy_val_dataset   = TensorDataset(val_features,   val_teacher_scores)

    proxy_train_loader = DataLoader(
        proxy_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,   # shuffle for SGD — important for convergence
    )
    proxy_val_loader = DataLoader(
        proxy_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Step 6 — Initialise proxy, loss, optimiser
    # ------------------------------------------------------------------
    proxy     = ProxyModel().to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(proxy.parameters(), lr=LR)

    best_agreement  = 0.0
    agreement_history = []   # track trajectory for debugging if needed
    Path("zkml/models").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 7 — Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):

        # --- Train ---
        proxy.train()
        train_loss = 0.0

        for features_batch, teacher_scores_batch in proxy_train_loader:
            features_batch       = features_batch.to(device)
            teacher_scores_batch = teacher_scores_batch.to(device)

            optimiser.zero_grad()
            proxy_scores = proxy(features_batch)
            loss         = criterion(proxy_scores, teacher_scores_batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(proxy_train_loader)

        # --- Evaluate ---
        proxy.eval()
        all_proxy_scores   = []
        all_teacher_scores = []

        with torch.no_grad():
            for features_batch, teacher_scores_batch in proxy_val_loader:
                features_batch = features_batch.to(device)
                scores         = proxy(features_batch).cpu()
                all_proxy_scores.append(scores)
                all_teacher_scores.append(teacher_scores_batch)

        all_proxy_scores   = torch.cat(all_proxy_scores)
        all_teacher_scores = torch.cat(all_teacher_scores)

        agreement = compute_agreement(all_proxy_scores, all_teacher_scores)
        agreement_history.append(agreement)

        logger.info(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Agreement: {agreement:.4f} "
            f"({'TARGET MET' if agreement >= AGREEMENT_TARGET else f'target: {AGREEMENT_TARGET:.0%}'})"
        )

        # --- Save best ---
        if agreement > best_agreement:
            best_agreement = agreement
            torch.save(proxy.state_dict(), PROXY_CHECKPOINT)
            logger.info(
                f"New best proxy saved — agreement: {best_agreement:.4f}"
            )

        # --- Early stopping ---
        if agreement >= AGREEMENT_TARGET:
            logger.info(
                f"Agreement target reached at epoch {epoch}. "
                f"Proxy training complete."
            )
            break

    # --- Final summary ---
    logger.info(
        f"Distillation complete — "
        f"best agreement: {best_agreement:.4f} | "
        f"circuit: {CIRCUIT_VERSION} | "
        f"checkpoint: {PROXY_CHECKPOINT}"
    )

    # Log agreement trajectory — useful if target was not reached
    if best_agreement < AGREEMENT_TARGET:
        logger.warning(
            f"Agreement {best_agreement:.4f} below target {AGREEMENT_TARGET}. "
            f"Trajectory: {[f'{a:.4f}' for a in agreement_history]}"
        )
        logger.warning(
            "Fixes to try in order: "
            "(1) increase EPOCHS, "
            "(2) lower LR to 5e-4, "
            "(3) verify feature extraction shape is [B, 128] (CrossAttentionFusion output)"
        )


if __name__ == "__main__":
    train()