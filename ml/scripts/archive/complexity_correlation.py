"""
complexity_correlation.py — Shortcut learning diagnostic.

Tests whether the model is using contract complexity (CFG node count, function
count, edge count) as a proxy for vulnerability instead of structural patterns.

Computes Spearman rank correlation between per-contract complexity metrics and
the model's predicted vulnerability probabilities across the val split.

If any correlation > 0.40: strong evidence of shortcut learning — the model is
using "this contract is large/complex" as its primary signal, not specific
vulnerability patterns. This puts a hard ceiling on improvement regardless of
architecture changes.

Usage:
    source ml/.venv/bin/activate
    python ml/scripts/complexity_correlation.py \\
        --checkpoint ml/checkpoints/v8.0-A-20260521_best.pt \\
        --cache ml/data/cached_dataset_deduped.pkl \\
        --splits-dir ml/data/splits/deduped \\
        --label-csv ml/data/processed/multilabel_index_cleaned.csv

Outputs:
    ml/logs/complexity_correlation_<run_name>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import numpy as np

# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "CallToUnknown",
    "DenialOfService",
    "ExternalBug",
    "GasException",
    "IntegerUO",
    "MishandledException",
    "Reentrancy",
    "Timestamp",
    "TransactionOrderDependence",
    "UnusedReturn",
]

EDGE_CF = 6  # CONTROL_FLOW edge type — CFG edges

# ---------------------------------------------------------------------------

def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy dependency."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return float(1.0 - 6.0 * (d * d).sum() / (n * (n * n - 1)))


def load_model(checkpoint_path: str, device: torch.device):
    """Load SentinelModel from checkpoint dict."""
    # Add project root (sentinel/) to path so 'ml.src.*' imports resolve
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from ml.src.models.sentinel_model import SentinelModel  # type: ignore

    raw = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = raw["config"]

    model = SentinelModel(
        num_classes            = cfg.get("num_classes", 10),
        fusion_output_dim      = cfg.get("fusion_output_dim", 128),
        gnn_hidden_dim         = cfg.get("gnn_hidden_dim", 256),
        gnn_num_layers         = cfg.get("gnn_layers", 7),
        gnn_heads              = cfg.get("gnn_heads", 8),
        gnn_dropout            = cfg.get("gnn_dropout", 0.2),
        use_edge_attr          = cfg.get("use_edge_attr", True),
        gnn_edge_emb_dim       = cfg.get("gnn_edge_emb_dim", 64),
        gnn_use_jk             = cfg.get("gnn_use_jk", True),
        gnn_jk_mode            = cfg.get("gnn_jk_mode", "attention"),
        gnn_phase2_edge_types  = cfg.get("gnn_phase2_edge_types", None),
        lora_r                 = cfg.get("lora_r", 16),
        lora_alpha             = cfg.get("lora_alpha", 32),
        lora_dropout           = cfg.get("lora_dropout", 0.1),
        lora_target_modules    = cfg.get("lora_target_modules", None),
    )
    # torch.compile() inserts '._orig_mod.' inside submodule key paths.
    # e.g. "gnn._orig_mod.conv1.weight" → "gnn.conv1.weight"
    # Strip all occurrences so the state dict loads into an uncompiled model.
    state_dict = raw["model"]
    if any("._orig_mod." in k for k in state_dict):
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    # Model was trained with BF16 AMP — cast to float32 for diagnostic inference
    model = model.float()
    return model.to(device)


def compute_complexity_metrics(graph) -> dict:
    """Extract per-contract complexity metrics from graph Data object."""
    num_nodes = graph.x.shape[0] if graph.x is not None else 0
    num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0

    ea = graph.edge_attr
    num_cfg_edges = 0
    if ea is not None:
        if ea.dim() > 1:
            ea = ea.squeeze(-1)
        num_cfg_edges = int((ea == EDGE_CF).sum().item())

    # external_call_count sum across all nodes (dim[10]) — unnormalized proxy
    if graph.x is not None:
        ext_calls_sum = float(graph.x[:, 10].sum().item())
        max_complexity = float(graph.x[:, 5].max().item())  # complexity dim[5]
    else:
        ext_calls_sum = 0.0
        max_complexity = 0.0

    return {
        "num_nodes":    num_nodes,
        "num_edges":    num_edges,
        "num_cfg_edges": num_cfg_edges,
        "ext_calls_sum": ext_calls_sum,
        "max_complexity": max_complexity,
    }


def run_inference(model, graph, token_dict, device: torch.device) -> np.ndarray:
    """Return sigmoid probabilities [10] for a single contract."""
    from torch_geometric.data import Batch  # type: ignore

    graph = graph.to(device)
    batch = Batch.from_data_list([graph])

    # Token dict: {input_ids, attention_mask} each [4, 512]
    input_ids      = token_dict["input_ids"].unsqueeze(0).to(device)   # [1, 4, 512]
    attention_mask = token_dict["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            graphs         = batch,
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    return probs


def main():
    parser = argparse.ArgumentParser(description="Complexity vs prediction correlation diagnostic")
    parser.add_argument("--checkpoint",  default="ml/checkpoints/v8.0-A-20260521_best.pt")
    parser.add_argument("--cache",       default="ml/data/cached_dataset_deduped.pkl")
    parser.add_argument("--splits-dir",  default="ml/data/splits/deduped")
    parser.add_argument("--label-csv",   default="ml/data/processed/multilabel_index_cleaned.csv")
    parser.add_argument("--max-contracts", type=int, default=2000,
                        help="Max val contracts to process (default: all, capped at 2000 for speed)")
    parser.add_argument("--output-dir",  default="ml/logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load val split md5 stems — supports both text (val.txt) and numpy (val_indices.npy) formats
    val_stems: set[str] = set()
    val_txt = Path(args.splits_dir) / "val.txt"
    val_npy = Path(args.splits_dir) / "val_indices.npy"

    if val_txt.exists():
        val_stems = set(val_txt.read_text().splitlines())
    elif val_npy.exists():
        import numpy as np
        import csv as csv_mod
        val_indices = np.load(str(val_npy))
        # Read the label CSV to get ordered stems; indices are positional into that list
        with open(args.label_csv, newline="") as f:
            all_stems = [row["md5_stem"] for row in csv_mod.DictReader(f)]
        val_stems = set(all_stems[i] for i in val_indices if i < len(all_stems))
        print(f"Loaded {len(val_indices)} val indices → {len(val_stems)} stems from CSV")
    else:
        print(f"ERROR: val split not found (tried {val_txt} and {val_npy})", file=sys.stderr)
        sys.exit(1)
    print(f"Val split: {len(val_stems)} contracts")

    # Load cache
    print(f"Loading cache: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    cache = {k: v for k, v in cache.items() if k != "__schema_version__"}
    print(f"Cache size: {len(cache)} entries")

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("Model loaded")

    # Intersect val stems with cache
    available = [s for s in val_stems if s in cache]
    if args.max_contracts and len(available) > args.max_contracts:
        import random; random.seed(42)
        available = random.sample(available, args.max_contracts)
    print(f"Processing {len(available)} val contracts")

    # Collect metrics
    records = []
    errors = 0
    from tqdm import tqdm  # type: ignore

    for stem in tqdm(available, desc="Inferring"):
        graph, token_dict = cache[stem]
        try:
            complexity = compute_complexity_metrics(graph)
            probs = run_inference(model, graph, token_dict, device)
            records.append({
                "stem":         stem,
                "complexity":   complexity,
                "probs":        probs.tolist(),
                "max_prob":     float(probs.max()),
                "mean_prob":    float(probs.mean()),
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  WARN: {stem}: {e}", file=sys.stderr)

    print(f"\nProcessed {len(records)} contracts, {errors} errors")

    # Compute correlations
    metrics_keys = ["num_nodes", "num_edges", "num_cfg_edges", "ext_calls_sum", "max_complexity"]
    pred_keys    = CLASS_NAMES + ["max_prob", "mean_prob"]

    # Build arrays
    metrics_arr = {k: np.array([r["complexity"][k] for r in records]) for k in metrics_keys}
    probs_arr   = {
        CLASS_NAMES[i]: np.array([r["probs"][i] for r in records])
        for i in range(len(CLASS_NAMES))
    }
    probs_arr["max_prob"]  = np.array([r["max_prob"]  for r in records])
    probs_arr["mean_prob"] = np.array([r["mean_prob"] for r in records])

    print("\n--- Spearman Correlation: Complexity Metrics vs Predicted Probability ---")
    print(f"{'Metric':<20} {'Class':<22} {'r':>8}  Interpretation")
    print("-" * 70)

    correlations = {}
    alerts = []
    for metric_k in metrics_keys:
        m = metrics_arr[metric_k]
        correlations[metric_k] = {}
        for pred_k in pred_keys:
            p = probs_arr[pred_k]
            r = spearman_r(m, p)
            correlations[metric_k][pred_k] = round(r, 4)
            flag = ""
            if abs(r) > 0.40:
                flag = "  *** SHORTCUT ALERT ***"
                alerts.append(f"{metric_k} vs {pred_k}: r={r:.3f}")
            elif abs(r) > 0.25:
                flag = "  (moderate)"
            print(f"  {metric_k:<18} {pred_k:<22} {r:>8.3f}{flag}")
        print()

    # Summary
    print("\n--- Summary ---")
    if alerts:
        print("SHORTCUT LEARNING EVIDENCE (r > 0.40):")
        for a in alerts:
            print(f"  {a}")
        print("\nInterpretation: The model is using contract complexity as a proxy.")
        print("Fix priority: label quality → per-statement features → JK cat mode.")
    else:
        print("No strong complexity shortcuts detected (r < 0.40 for all metrics).")
        print("The model is likely learning structural patterns, not raw complexity.")

    # Save report
    run_name = Path(args.checkpoint).stem
    output_path = Path(args.output_dir) / f"complexity_correlation_{run_name}.json"
    with open(output_path, "w") as f:
        json.dump({
            "checkpoint":    args.checkpoint,
            "num_contracts": len(records),
            "correlations":  correlations,
            "alerts":        alerts,
        }, f, indent=2)
    print(f"\nFull report saved: {output_path}")


if __name__ == "__main__":
    main()
