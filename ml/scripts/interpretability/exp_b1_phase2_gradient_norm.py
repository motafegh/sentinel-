"""
exp_b1_phase2_gradient_norm.py — B1: Phase 2 Gradient Norm per LayerNorm Output

PURPOSE
───────
Measure the gradient norm at each phase's LayerNorm output during a supervised
forward pass.  A phase with near-zero gradient norm is not receiving useful
loss signal — indicating the model has collapsed to ignoring that phase.

Key questions answered:
- Does Phase 2 (CF/ICFG) receive meaningful gradient signal for Reentrancy?
- Is Phase 2 gradient disproportionately lower than Phase 1/3?
- Which classes provide the strongest Phase 2 gradient (i.e. which classes
  actually benefit from CFG structure)?

APPROACH
─────────
1. Register backward hooks on the three phase LayerNorms:
     gnn.ln_phase1, gnn.ln_phase2, gnn.ln_phase3
2. For each class C and N_SAMPLES val-split positive contracts:
   a. Forward pass → logits[0, C] (no sigmoid, logit space)
   b. backward()
   c. Record the L2 norm of grad at each phase LayerNorm output
3. Aggregate: mean ± std per phase per class

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_b1_phase2_gradient_norm.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --cache ml/data/cached_dataset_v10.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --out ml/logs/interpretability/b1_phase2_gradient_norm.json

OUTPUT
──────
    - Per-class per-phase gradient norm table (stdout)
    - Heatmap PNG: class × phase mean gradient norms
    - JSON report at --out

EXIT CODES
──────────
    0  completed (even if Phase 2 looks weak)
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
    CLASS_NAMES,
    PHASE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

N_SAMPLES_PER_CLASS = 20


# ── Gradient norm measurement ─────────────────────────────────────────────────

def measure_phase_gradient_norms(
    model,
    stems: list[str],
    df_split,
    cache: dict,
    device: str,
) -> dict:
    """
    For each class, measure grad norms at phase LayerNorm outputs.

    Returns:
        {class_name: {"phase1": [norm, ...], "phase2": [...], "phase3": [...]}}
    """
    from torch_geometric.data import Batch

    gnn = model.gnn
    # GNNEncoder uses phase_norm ModuleList: [0]=after Phase1, [1]=after Phase2, [2]=after Phase3
    ln_phase1 = gnn.phase_norm[0]
    ln_phase2 = gnn.phase_norm[1]
    ln_phase3 = gnn.phase_norm[2]

    stem_to_labels = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    results: dict = {cls: {"phase1": [], "phase2": [], "phase3": []} for cls in CLASS_NAMES}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        pos_stems = [s for s in stems if s in stem_to_labels
                     and stem_to_labels[s][class_idx] == 1 and s in cache]
        if not pos_stems:
            log.warning(f"  {class_name}: no positive stems in val split")
            continue

        rng = np.random.default_rng(42)
        if len(pos_stems) > N_SAMPLES_PER_CLASS:
            pos_stems = rng.choice(pos_stems, size=N_SAMPLES_PER_CLASS, replace=False).tolist()

        log.info(f"  {class_name}: {len(pos_stems)} samples")

        for stem in pos_stems:
            entry = cache[stem]
            if not isinstance(entry, tuple):
                continue
            graph, token = entry

            try:
                batch     = Batch.from_data_list([graph]).to(device)
                input_ids = token["input_ids"].unsqueeze(0).to(device)
                attn_mask = token["attention_mask"].unsqueeze(0).to(device)

                # Enable grad on GNN parameters
                for p in gnn.parameters():
                    p.requires_grad_(True)

                captured_grads: dict = {}

                def make_hook(key):
                    def hook(grad):
                        captured_grads[key] = grad.detach().cpu()
                    return hook

                # Register hooks on LayerNorm OUTPUT (input to next phase)
                # We hook the inputs by registering on the module outputs
                handles = []
                for layer_name, ln_module in [
                    ("phase1", ln_phase1),
                    ("phase2", ln_phase2),
                    ("phase3", ln_phase3),
                ]:
                    # hook via a dummy parameter: instead hook the GNN params
                    pass

                # Approach: use register_hook on each LN module output tensor
                # via a forward hook that saves the output tensor then
                # calls retain_grad() on it.
                saved_outputs: dict = {}

                def make_fwd_hook(key):
                    def fwd_hook(module, inp, out):
                        saved_outputs[key] = out
                        out.retain_grad()
                    return fwd_hook

                h1 = ln_phase1.register_forward_hook(make_fwd_hook("phase1"))
                h2 = ln_phase2.register_forward_hook(make_fwd_hook("phase2"))
                h3 = ln_phase3.register_forward_hook(make_fwd_hook("phase3"))

                model.train()
                logits, _ = model(batch, input_ids, attn_mask, return_aux=True)
                # Use BCEWithLogitsLoss on positive target so gradient magnitude
                # matches training-time behavior (sigmoid + BCE), not raw logit.
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[0, class_idx].unsqueeze(0),
                    torch.ones(1, device=logits.device),
                )
                loss.backward()
                model.eval()

                h1.remove(); h2.remove(); h3.remove()

                for phase_key in ("phase1", "phase2", "phase3"):
                    tensor = saved_outputs.get(phase_key)
                    if tensor is not None and tensor.grad is not None:
                        norm = float(tensor.grad.norm().item())
                        results[class_name][phase_key].append(norm)
                    else:
                        results[class_name][phase_key].append(0.0)

                # Zero grads for next iteration
                for p in gnn.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            except Exception as exc:
                log.debug(f"  Skipping {stem}: {exc}")

    return results


# ── Aggregation and reporting ─────────────────────────────────────────────────

def summarise(results: dict) -> dict:
    summary = {}
    for cls, phases in results.items():
        summary[cls] = {}
        for phase, norms in phases.items():
            if norms:
                summary[cls][phase] = {
                    "mean": round(float(np.mean(norms)), 6),
                    "std":  round(float(np.std(norms)), 6),
                    "n":    len(norms),
                }
            else:
                summary[cls][phase] = {"mean": 0.0, "std": 0.0, "n": 0}
    return summary


def _print_table(summary: dict) -> None:
    phases = ["phase1", "phase2", "phase3"]
    header = f"{'Class':26s}" + "".join(f"  {p:>12s}" for p in phases)
    print(f"\n{'═'*70}")
    print("  B1: Phase 2 Gradient Norm per Class")
    print(f"{'═'*70}")
    print(f"  {header}")
    print(f"  {'-'*68}")
    for cls in CLASS_NAMES:
        row = f"  {cls:26s}"
        for ph in phases:
            m = summary.get(cls, {}).get(ph, {}).get("mean", 0.0)
            row += f"  {m:>12.6f}"
        print(row)
    print(f"{'═'*70}\n")


def _save_heatmap(summary: dict, out_dir: Path) -> None:
    phases = ["phase1", "phase2", "phase3"]
    matrix = np.array([
        [summary.get(cls, {}).get(ph, {}).get("mean", 0.0) for ph in phases]
        for cls in CLASS_NAMES
    ])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Mean grad norm")
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(["Phase 1\n(struct+CONTAINS)", "Phase 2\n(CF/ICFG/DFG)", "Phase 3\n(rev-CONTAINS)"], fontsize=9)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    for i, cls in enumerate(CLASS_NAMES):
        for j, ph in enumerate(phases):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7,
                    color="white" if val > matrix.max() * 0.6 else "black")
    ax.set_title("B1: Phase Gradient Norms per Class\n(GNN LayerNorm output)")
    plt.tight_layout()
    out_path = out_dir / "b1_phase_gradient_norms.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Heatmap saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B1: Phase 2 Gradient Norm Analysis")
    add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/b1_phase2_gradient_norm.json")
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

    log.info("Measuring phase gradient norms...")
    results = measure_phase_gradient_norms(model, stems, df_split, cache, args.device)
    summary = summarise(results)

    _print_table(summary)
    _save_heatmap(summary, out_path.parent)

    report = {
        "experiment": "exp_b1_phase2_gradient_norm",
        "checkpoint": str(args.checkpoint),
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "summary": summary,
        "raw": {cls: {ph: norms for ph, norms in phases.items()}
                for cls, phases in results.items()},
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
