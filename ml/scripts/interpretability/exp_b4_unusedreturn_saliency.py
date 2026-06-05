"""
exp_b4_unusedreturn_saliency.py — B4: UnusedReturn Top-Scored Gradient Saliency

PURPOSE
───────
EXP-L6 showed UnusedReturn was the ONLY counterfactual PASS (score dropped
+0.108 when the vulnerable contract was replaced with a safe one).  This
suggests the model has genuine structural signal for UnusedReturn — but we
don't know whether it uses the `return_ignored` feature (dim 7) or just
responds to size/complexity shortcuts.

This experiment takes the TOP-N highest UnusedReturn-scored contracts from the
val split and runs gradient saliency to answer:
- Is `return_ignored` (dim 7) in the top-3 most salient features?
- Or is the dominant feature `complexity` (dim 5) / `loc` (dim 6)
  suggesting a size shortcut?
- Compare against bottom-N lowest-scored positive contracts.

APPROACH
─────────
1. Collect UnusedReturn logit scores for all val-split positive contracts.
2. Take top-N by score (model is most confident) and bottom-N by score.
3. For each contract, compute gradient saliency (EXP-L4 method):
   x.requires_grad_(True) → forward → logits[0, UnusedReturn_idx].backward()
4. Aggregate saliency per feature dim; report which dims dominate.

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_b4_unusedreturn_saliency.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --cache ml/data/cached_dataset_v10.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --out ml/logs/interpretability/b4_unusedreturn_saliency.json

OUTPUT
──────
    - Feature saliency ranking for top-N and bottom-N contracts (stdout)
    - Bar chart: top-N vs bottom-N mean saliency per feature dim (PNG)
    - JSON report at --out

EXIT CODES
──────────
    0  completed
    1  fatal error
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ml.scripts.interpretability.utils import (
    load_model,
    load_val_split,
    add_common_args,
    collect_predictions,
    CLASS_NAMES,
)
from ml.src.preprocessing.graph_schema import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

_UNUSED_RETURN_IDX = CLASS_NAMES.index("UnusedReturn")
N_TOP = 30
N_DIM = len(FEATURE_NAMES)  # 11


# ── Gradient saliency for one contract ────────────────────────────────────────

def _saliency_one(model, graph, token, class_idx: int, device: str) -> np.ndarray | None:
    """
    Compute per-feature gradient saliency for one contract.

    Returns:
        [N_DIM] float array — mean |∂logit/∂x| per feature dim, or None on failure.
    """
    from torch_geometric.data import Batch

    try:
        batch     = Batch.from_data_list([graph]).to(device)
        input_ids = token["input_ids"].unsqueeze(0).to(device)
        attn_mask = token["attention_mask"].unsqueeze(0).to(device)

        x = batch.x.float().requires_grad_(True)
        batch.x = x

        model.train()
        logits, _ = model(batch, input_ids, attn_mask, return_aux=True)
        logits[0, class_idx].backward()
        model.eval()

        if x.grad is None:
            return None

        saliency = x.grad.abs().detach().cpu()  # [N_nodes, 11]
        return saliency.mean(dim=0).numpy()     # [11]
    except Exception as exc:
        log.debug(f"  Saliency failed: {exc}")
        model.eval()
        return None


# ── Collection ────────────────────────────────────────────────────────────────

def collect_saliencies(
    model,
    stems: list[str],
    df_split,
    cache: dict,
    device: str,
    preds_logits: np.ndarray,
    pred_stems: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Collect gradient saliencies for top-N and bottom-N UnusedReturn-positive contracts.

    Returns:
        top_sal [N_TOP, 11], bot_sal [N_TOP, 11], top_stems, bot_stems
    """
    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    # Map stem → UnusedReturn logit score
    stem_idx = {s: i for i, s in enumerate(pred_stems)}
    ur_pos_stems = [
        s for s in pred_stems
        if s in stem_to_labels and stem_to_labels[s][_UNUSED_RETURN_IDX] == 1
    ]

    if not ur_pos_stems:
        raise RuntimeError("No UnusedReturn positive contracts found in predictions.")

    ur_scores = [(s, float(preds_logits[stem_idx[s], _UNUSED_RETURN_IDX])) for s in ur_pos_stems]
    ur_scores.sort(key=lambda x: -x[1])

    top_stems_raw = [s for s, _ in ur_scores[:N_TOP]]
    bot_stems_raw = [s for s, _ in ur_scores[-N_TOP:]]

    def _run(stem_list: list[str]) -> tuple[np.ndarray, list[str]]:
        sals = []
        valid = []
        for stem in stem_list:
            if stem not in cache:
                continue
            entry = cache[stem]
            if not isinstance(entry, tuple):
                continue
            graph, token = entry
            sal = _saliency_one(model, graph, token, _UNUSED_RETURN_IDX, device)
            if sal is not None:
                sals.append(sal)
                valid.append(stem)
        return (np.stack(sals) if sals else np.zeros((0, N_DIM))), valid

    log.info(f"Computing saliency for top-{N_TOP} UnusedReturn contracts...")
    top_sal, top_stems = _run(top_stems_raw)
    log.info(f"Computing saliency for bottom-{N_TOP} UnusedReturn contracts...")
    bot_sal, bot_stems = _run(bot_stems_raw)

    return top_sal, bot_sal, top_stems, bot_stems


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_ranking(label: str, sal_matrix: np.ndarray) -> list[dict]:
    if sal_matrix.shape[0] == 0:
        log.warning(f"No saliency data for {label}")
        return []
    mean_sal = sal_matrix.mean(axis=0)
    order = np.argsort(-mean_sal)
    print(f"\n  {label} — Mean |∂logit/∂x| per feature dim (n={sal_matrix.shape[0]})")
    print(f"  {'Rank':>5}  {'Dim':>4}  {'Feature':25s}  {'Saliency':>10}")
    print(f"  {'-'*55}")
    rows = []
    for rank, dim in enumerate(order):
        print(f"  {rank+1:>5}  {dim:>4}  {FEATURE_NAMES[dim]:25s}  {mean_sal[dim]:>10.6f}")
        rows.append({"rank": rank + 1, "dim": int(dim), "feature": FEATURE_NAMES[dim],
                     "mean_saliency": round(float(mean_sal[dim]), 6)})
    return rows


def _save_comparison_bar(top_sal: np.ndarray, bot_sal: np.ndarray, out_dir: Path) -> None:
    if top_sal.shape[0] == 0 or bot_sal.shape[0] == 0:
        return
    top_mean = top_sal.mean(axis=0)
    bot_mean = bot_sal.mean(axis=0)
    x = np.arange(N_DIM)
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, top_mean, width, label=f"Top-{N_TOP} (highest score)", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, bot_mean, width, label=f"Bottom-{N_TOP} (lowest score)", color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{i}]\n{fn[:8]}" for i, fn in enumerate(FEATURE_NAMES)],
                       fontsize=7, ha="center")
    ax.set_ylabel("Mean |∂logit/∂x|")
    ax.set_title("B4: UnusedReturn Gradient Saliency — Top vs Bottom Scored Contracts\n"
                 "dim 7 = return_ignored (UnusedReturn signal expected)")
    ax.axvline(6.5, color="gray", linestyle=":", linewidth=0.8)
    ax.text(7, ax.get_ylim()[1] * 0.9, "← return_ignored (7)", fontsize=8, color="darkgreen")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / "b4_unusedreturn_saliency.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Bar chart saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B4: UnusedReturn Top-Scored Gradient Saliency")
    add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/b4_unusedreturn_saliency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stems, df_split, cache = load_val_split(
            cache_path = Path(args.cache),
            label_csv  = Path(args.label_csv),
            splits_dir = Path(args.splits_dir),
            split      = args.split,
        )
    except FileNotFoundError as exc:
        log.error(str(exc))
        return 1

    try:
        model = load_model(checkpoint_path=Path(args.checkpoint), device=args.device)
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        return 1

    log.info("Collecting UnusedReturn predictions to rank contracts...")
    preds = collect_predictions(
        model=model, stems=stems, df_split=df_split, cache=cache,
        device=args.device, return_aux=False,
    )

    print(f"\n{'═'*60}")
    print("  B4: UnusedReturn Gradient Saliency")
    print(f"{'═'*60}")

    try:
        top_sal, bot_sal, top_stems, bot_stems = collect_saliencies(
            model=model, stems=stems, df_split=df_split, cache=cache,
            device=args.device,
            preds_logits=preds["logits"],
            pred_stems=preds["stems"],
        )
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    top_ranking = _print_ranking(f"Top-{len(top_stems)} highest-scored contracts", top_sal)
    bot_ranking = _print_ranking(f"Bottom-{len(bot_stems)} lowest-scored contracts", bot_sal)
    _save_comparison_bar(top_sal, bot_sal, out_path.parent)

    # Key diagnostic: is dim 7 (return_ignored) in top-3 for highest-scored?
    if top_ranking:
        top3_dims = [r["dim"] for r in top_ranking[:3]]
        return_ignored_rank = next((r["rank"] for r in top_ranking if r["dim"] == 7), None)
        print(f"\n  Key finding:")
        print(f"    return_ignored (dim 7) rank in top-scored contracts: {return_ignored_rank}")
        if return_ignored_rank is not None and return_ignored_rank <= 3:
            print("    → STRUCTURAL SIGNAL: model uses return_ignored feature for UnusedReturn")
        else:
            print("    → SIZE SHORTCUT: model NOT using return_ignored — likely complexity/loc shortcut")

    print(f"\n{'═'*60}\n")

    report = {
        "experiment": "exp_b4_unusedreturn_saliency",
        "checkpoint": str(args.checkpoint),
        "n_top": len(top_stems),
        "n_bottom": len(bot_stems),
        "feature_names": list(FEATURE_NAMES),
        "top_scored": {
            "stems": top_stems,
            "mean_saliency_per_dim": top_sal.mean(axis=0).tolist() if top_sal.shape[0] > 0 else [],
            "ranking": top_ranking,
        },
        "bottom_scored": {
            "stems": bot_stems,
            "mean_saliency_per_dim": bot_sal.mean(axis=0).tolist() if bot_sal.shape[0] > 0 else [],
            "ranking": bot_ranking,
        },
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
