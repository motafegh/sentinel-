"""
jk_weight_hist.py — JK attention weight distribution diagnostic.

The training log reports one number per phase per epoch (the mean across all
nodes in the last training batch).  That mean collapses all per-node variance.
This script answers the question the mean cannot:

  Is Phase 3 dominating because every node genuinely prefers Phase 3,
  or because Phase 1 / Phase 2 oversmoothed and left Phase 3 as the only
  phase with useful variance?

How it works
────────────
Load a checkpoint, run N contracts through the GNN in eval mode, collect the
full [N_nodes, 3] weight tensor that _JKAttention stores in last_node_weights,
and report:

  • Per-phase mean, std, p5, p25, p50, p75, p95
  • Fraction of nodes where each phase is the argmax (dominant phase count)
  • Entropy of the weight distribution per node, averaged across all nodes
    (low entropy → near-one-hot per node; high entropy → near-uniform)
  • ASCII histogram of Phase 1 / Phase 2 / Phase 3 weight distributions

Interpretation guide
────────────────────
  Narrow histogram + std < 0.05:   per-node mechanism collapsed to global weights.
                                    The learned attention is functionally uniform.
  Wide histogram + std > 0.15:     genuine per-node specialization is happening.
  Bimodal histogram:               two distinct node populations exist (e.g. CFG
                                    nodes prefer Phase 2, AST nodes prefer Phase 3).
  Mean ≈ reported training mean:   eval and train batches are representative.
  Mean >> reported training mean:  last training batch was unrepresentative; the
                                    epoch-level JK numbers should not be trusted
                                    as a stable signal.

Usage
─────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/jk_weight_hist.py \\
        --checkpoint ml/checkpoints/v8.0-AB-20260520_best.pt \\
        --cache-path ml/data/cached_dataset_v8.pkl \\
        --splits-dir ml/data/splits/deduped \\
        --split val \\
        --n-contracts 1000 \\
        --out ml/logs/jk_hist_v8AB.json

    # Compare two checkpoints side by side:
    PYTHONPATH=. python ml/scripts/jk_weight_hist.py --checkpoint ml/checkpoints/v7.0_best.pt ...
    PYTHONPATH=. python ml/scripts/jk_weight_hist.py --checkpoint ml/checkpoints/v8.0-AB-20260520_best.pt ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from ml.src.models.sentinel_model import SentinelModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PHASE_NAMES = ["Phase1 (struct+CONTAINS)", "Phase2 (CF/ICFG/DFG)", "Phase3 (rev-CONTAINS)"]


# ── checkpoint loading (same fixes as tune_threshold.py) ─────────────────────

def _load_model(checkpoint: Path, device: str, phase2_edge_types: list[int] | None) -> SentinelModel:
    log.info(f"Loading checkpoint: {checkpoint}")
    raw = torch.load(checkpoint, map_location=device, weights_only=False)

    # Checkpoint is a dict with 'model' state_dict and 'config' arch params
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    ckpt_cfg   = raw.get("config", {}) if isinstance(raw, dict) else {}

    # Strip torch.compile _orig_mod. prefix
    state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

    def _ensure_list(v, default):
        if v is None: return default
        return list(v)

    model = SentinelModel(
        num_classes       = int(ckpt_cfg.get("num_classes", 10)),
        fusion_output_dim = int(ckpt_cfg.get("fusion_output_dim", 128)),
        dropout           = float(ckpt_cfg.get("fusion_dropout", 0.3)),
        gnn_hidden_dim    = int(ckpt_cfg.get("gnn_hidden_dim", 256)),
        gnn_num_layers    = int(ckpt_cfg.get("gnn_layers", 7)),
        gnn_heads         = int(ckpt_cfg.get("gnn_heads", 8)),
        gnn_dropout       = float(ckpt_cfg.get("gnn_dropout", 0.2)),
        use_edge_attr     = bool(ckpt_cfg.get("use_edge_attr", True)),
        gnn_edge_emb_dim  = int(ckpt_cfg.get("gnn_edge_emb_dim", 64)),
        gnn_use_jk        = bool(ckpt_cfg.get("gnn_use_jk", True)),
        gnn_jk_mode       = str(ckpt_cfg.get("gnn_jk_mode", "attention")),
        # phase2_edge_types: CLI override takes precedence over checkpoint config
        gnn_phase2_edge_types = phase2_edge_types if phase2_edge_types is not None
                                else ckpt_cfg.get("gnn_phase2_edge_types"),
        lora_r            = int(ckpt_cfg.get("lora_r", 16)),
        lora_alpha        = int(ckpt_cfg.get("lora_alpha", 32)),
        lora_dropout      = float(ckpt_cfg.get("lora_dropout", 0.1)),
        lora_target_modules = _ensure_list(
            ckpt_cfg.get("lora_target_modules"), ["query", "value"]
        ),
    ).to(device)

    # Resize edge embedding if checkpoint was saved with different NUM_EDGE_TYPES
    edge_emb_key = next((k for k in state_dict if "edge_embedding.weight" in k), None)
    if edge_emb_key and model.gnn.edge_embedding is not None:
        ckpt_n = state_dict[edge_emb_key].shape[0]
        curr_n = model.gnn.edge_embedding.num_embeddings
        if ckpt_n != curr_n:
            emb_dim = model.gnn.edge_embedding.embedding_dim
            model.gnn.edge_embedding = nn.Embedding(ckpt_n, emb_dim).to(device)
            log.info(f"Resized edge_embedding: {curr_n} → {ckpt_n}")

    model.load_state_dict(state_dict, strict=False)
    model.float()   # normalise BF16 AMP checkpoints
    model.eval()
    return model


# ── data loading ─────────────────────────────────────────────────────────────

def _load_samples(
    cache_path: Path,
    splits_dir: Path,
    label_csv: Path,
    split: str,
    n: int,
    seed: int,
) -> list[tuple]:
    """Return up to n (graph, token) pairs from the split."""
    log.info(f"Loading cache: {cache_path}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    df = pd.read_csv(label_csv)
    indices = np.load(splits_dir / f"{split}_indices.npy")
    stems = df["md5_stem"].iloc[indices].tolist()

    # Subsample reproducibly
    rng = np.random.default_rng(seed)
    if len(stems) > n:
        stems = rng.choice(stems, size=n, replace=False).tolist()

    pairs = []
    for stem in stems:
        if stem in cache and isinstance(cache[stem], tuple):
            pairs.append(cache[stem])
    log.info(f"Loaded {len(pairs)} samples from split='{split}'")
    return pairs


# ── collection pass ───────────────────────────────────────────────────────────

def _collect_node_weights(
    model: SentinelModel,
    samples: list[tuple],
    device: str,
) -> np.ndarray:
    """
    Run each sample through the model in eval mode and collect per-node JK
    weights from model.gnn.jk.last_node_weights after each forward pass.

    Returns ndarray of shape [total_nodes, 3].
    """
    from torch_geometric.data import Batch

    all_weights: list[np.ndarray] = []

    with torch.no_grad():
        for graph, token in samples:
            try:
                batch = Batch.from_data_list([graph]).to(device)
                input_ids      = token["input_ids"].unsqueeze(0).to(device)
                attention_mask = token["attention_mask"].unsqueeze(0).to(device)

                _ = model(batch, input_ids, attention_mask)

                node_w = model.gnn.jk.last_node_weights  # [N, 3] on CPU
                if node_w is not None:
                    all_weights.append(node_w.numpy())
            except Exception as exc:
                log.debug(f"Skipping sample: {exc}")
                continue

    if not all_weights:
        raise RuntimeError("No node weights collected — is gnn_use_jk=True?")

    result = np.concatenate(all_weights, axis=0)   # [total_nodes, 3]
    log.info(f"Collected weights for {result.shape[0]:,} nodes across {len(all_weights)} graphs")
    return result


# ── analysis ─────────────────────────────────────────────────────────────────

def _analyse(weights: np.ndarray) -> dict:
    """Compute per-phase stats and entropy from [N, 3] weight array."""
    N, K = weights.shape
    stats: dict = {"n_nodes": N, "n_phases": K, "phases": {}}

    for k in range(K):
        w = weights[:, k]
        stats["phases"][k] = {
            "name":  PHASE_NAMES[k],
            "mean":  float(np.mean(w)),
            "std":   float(np.std(w)),
            "p5":    float(np.percentile(w, 5)),
            "p25":   float(np.percentile(w, 25)),
            "p50":   float(np.percentile(w, 50)),
            "p75":   float(np.percentile(w, 75)),
            "p95":   float(np.percentile(w, 95)),
            "pct_dominant": float((np.argmax(weights, axis=1) == k).mean()),
        }

    # Per-node entropy: H = -sum(w * log(w)) for each node, then average
    eps = 1e-9
    entropy_per_node = -np.sum(weights * np.log(weights + eps), axis=1)
    max_entropy = math.log(K)
    stats["mean_entropy"]         = float(np.mean(entropy_per_node))
    stats["max_possible_entropy"] = float(max_entropy)
    stats["normalised_entropy"]   = float(np.mean(entropy_per_node) / max_entropy)

    return stats


def _ascii_histogram(values: np.ndarray, n_bins: int = 20, width: int = 40) -> list[str]:
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    max_count = max(counts) or 1
    lines = []
    for i, (c, lo) in enumerate(zip(counts, edges)):
        bar = "█" * int(width * c / max_count)
        lines.append(f"  {lo:.2f}-{edges[i+1]:.2f} | {bar:<{width}} {c:>6,}")
    return lines


def _print_report(stats: dict, checkpoint_name: str) -> None:
    print(f"\n{'═'*72}")
    print(f"  JK Weight Distribution — {checkpoint_name}")
    print(f"  Nodes analysed: {stats['n_nodes']:,}")
    print(f"{'═'*72}")

    print(f"\n  Per-phase summary:")
    print(f"  {'Phase':<30}  {'Mean':>6}  {'Std':>6}  {'p5':>5}  {'p50':>5}  {'p95':>5}  {'Dominant%':>9}")
    print(f"  {'-'*30}  {'------':>6}  {'------':>6}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'---------':>9}")
    for k, ph in stats["phases"].items():
        print(
            f"  {ph['name']:<30}  {ph['mean']:>6.3f}  {ph['std']:>6.3f}  "
            f"{ph['p5']:>5.3f}  {ph['p50']:>5.3f}  {ph['p95']:>5.3f}  "
            f"{ph['pct_dominant']:>8.1%}"
        )

    ne = stats["normalised_entropy"]
    print(f"\n  Mean entropy: {stats['mean_entropy']:.4f}  "
          f"(normalised: {ne:.3f} of max {stats['max_possible_entropy']:.4f})")

    if ne < 0.30:
        interp = "LOW — per-node mechanism is near one-hot; nodes strongly prefer one phase"
    elif ne < 0.60:
        interp = "MODERATE — some per-node specialization; not fully collapsed"
    else:
        interp = "HIGH — weights near-uniform per node; JK attention provides little selectivity"
    print(f"  Interpretation: {interp}")

    # std-based collapse diagnosis
    stds = [stats["phases"][k]["std"] for k in stats["phases"]]
    if max(stds) < 0.05:
        print("\n  ⚠ COLLAPSE SIGNAL: all per-phase std < 0.05 — per-node attention has")
        print("    collapsed to global weights. The mean JK log is the full story.")
    elif max(stds) > 0.15:
        print("\n  ✓ SPECIALIZATION SIGNAL: max std > 0.15 — genuine per-node variation present.")
        print("    The mean JK log understates what the model has learned.")

    print(f"\n{'─'*72}")


def _print_histograms(weights: np.ndarray) -> None:
    print(f"\n  Weight histograms (0.0–1.0, 20 bins):")
    for k, name in enumerate(PHASE_NAMES):
        print(f"\n  {name}:")
        for line in _ascii_histogram(weights[:, k]):
            print(line)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JK attention weight distribution diagnostic"
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to model checkpoint .pt")
    p.add_argument("--cache-path",   default="ml/data/cached_dataset_v8.pkl")
    p.add_argument("--label-csv",    default="ml/data/processed/multilabel_index_cleaned.csv")
    p.add_argument("--splits-dir",   default="ml/data/splits/deduped")
    p.add_argument("--split",        default="val", choices=["train", "val", "test"])
    p.add_argument("--n-contracts",  type=int, default=1000)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--phase2-edge-types", type=int, nargs="+", default=None,
                   dest="phase2_edge_types",
                   help="Override Phase 2 edge types (e.g. 6 8 9 for PLAN-3A). "
                        "Must match the edge types used during training.")
    p.add_argument("--out",          default=None,
                   help="Path to write JSON report (default: ml/logs/jk_hist_<ckpt_stem>.json)")
    p.add_argument("--no-hist",      action="store_true",
                   help="Skip ASCII histograms (useful for large N)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path  = Path(args.checkpoint)
    cache_path = Path(args.cache_path)
    splits_dir = Path(args.splits_dir)
    label_csv  = Path(args.label_csv)
    out_path   = Path(args.out) if args.out else Path(f"ml/logs/jk_hist_{ckpt_path.stem}.json")

    for p, name in [(ckpt_path, "checkpoint"), (cache_path, "cache"), (splits_dir, "splits_dir")]:
        if not p.exists():
            log.error(f"{name} not found: {p}")
            return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model   = _load_model(ckpt_path, args.device, args.phase2_edge_types)
    samples = _load_samples(cache_path, splits_dir, label_csv, args.split,
                            args.n_contracts, args.seed)
    weights = _collect_node_weights(model, samples, args.device)
    stats   = _analyse(weights)

    _print_report(stats, ckpt_path.name)
    if not args.no_hist:
        _print_histograms(weights)

    report = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_contracts": len(samples),
        "phase2_edge_types": args.phase2_edge_types,
        **stats,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
