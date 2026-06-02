"""
exp_l5_probing_classifiers.py — Layer 3, P1: Probing Classifiers Per Phase

PURPOSE
───────
Extract GNNEncoder per-phase node embeddings (after_phase1, after_phase2,
after_phase3) via return_intermediates=True, pool them per graph, then train
linear probes (logistic regression) to test how much linearly-separable
vulnerability signal each phase contains.

The Phase2-Phase1 F1 delta reveals whether the CONTROL_FLOW + ICFG layers
add genuinely new signal beyond the structural Phase 1 embedding.

LAYER / PRIORITY
─────────────────
Layer 3, Priority 1 — Representation quality per phase.

APPROACH
─────────
Step 1: For each val contract, run GNNEncoder.forward(..., return_intermediates=True)
        and pool each phase embedding over function-level nodes (types 1,2,4,5,6).
        Fallback: mean pool over all nodes if no function-level nodes exist.

Step 2: For each phase × class:
        - LogisticRegression probe (C=1.0, max_iter=500, solver='lbfgs')
        - 80/20 train/test split (stratified where possible)
        - Compute F1 (threshold=0.5) and AUROC on test set

Step 3: Report phase2-phase1 F1 delta per class.

PASS CRITERIA
─────────────
- Reentrancy: Phase2 probe F1 > Phase1 probe F1 + 3pp
- If Phase2 probe ≈ Phase1 probe for Reentrancy:
  → "CFG/ICFG phase adds no linearly-separable signal for reentrancy"

HOW TO RUN
──────────
    source ml/.venv/bin/activate
    PYTHONPATH=. python ml/scripts/interpretability/exp_l5_probing_classifiers.py \\
        --checkpoint ml/checkpoints/sentinel_best.pt \\
        --cache ml/data/cached_dataset_v9.pkl \\
        --label-csv ml/data/processed/multilabel_index.csv \\
        --splits-dir ml/data/splits/v9_deduped \\
        --n-contracts 2000 \\
        --out ml/logs/interpretability/l5_probing_classifiers.json

OUTPUT
──────
    - 3×10 F1 table printed to stdout (phase × class)
    - Phase2-Phase1 delta table
    - JSON report
    - (No PNG: tables are the primary output)

EXIT CODES
──────────
    0  analysis completed
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
    get_node_type_tensor,
    plot_class_heatmap,
)
from ml.src.preprocessing.graph_schema import NODE_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MAX_TYPE_ID: float = float(max(NODE_TYPES.values()))  # 12.0

# Function-level node type IDs (same set as SentinelModel uses for GNN eye pooling)
_FUNC_TYPE_IDS: set[int] = {
    NODE_TYPES["FUNCTION"],     # 1
    NODE_TYPES["MODIFIER"],     # 2
    NODE_TYPES["FALLBACK"],     # 4
    NODE_TYPES["RECEIVE"],      # 5
    NODE_TYPES["CONSTRUCTOR"],  # 6
}
_FUNC_TYPE_TENSOR: torch.Tensor = torch.tensor(sorted(_FUNC_TYPE_IDS), dtype=torch.long)

_PHASE_KEYS: list[str] = ["after_phase1", "after_phase2", "after_phase3"]


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_phase_embeddings(
    model,
    stems: list[str],
    cache: dict,
    df_split,
    device: str,
    max_contracts: int,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """
    Extract pooled per-phase GNN embeddings for up to max_contracts val contracts.

    Returns:
        embeddings: {"after_phase1": [N, 512], "after_phase2": ..., "after_phase3": ...}
                    (max+mean pool of function-level nodes → [512], matching sentinel_model)
        labels:     [N, 10] binary label matrix
        valid_stems: list of stems in order
    """
    from torch_geometric.data import Batch

    # Subsample if needed
    rng = np.random.default_rng(seed)
    if max_contracts is not None and len(stems) > max_contracts:
        idx = rng.choice(len(stems), size=max_contracts, replace=False)
        stems_sub = [stems[i] for i in idx]
    else:
        stems_sub = list(stems)

    stem_to_label = {
        row["md5_stem"]: [int(row[c]) for c in CLASS_NAMES]
        for _, row in df_split.iterrows()
    }

    gnn = model.gnn  # GNNEncoder

    phase_embs: dict[str, list[np.ndarray]] = {k: [] for k in _PHASE_KEYS}
    all_labels: list[list[int]] = []
    valid_stems: list[str] = []

    model.eval()
    skipped = 0

    for stem in stems_sub:
        if stem not in cache:
            skipped += 1
            continue
        entry = cache[stem]
        if not isinstance(entry, tuple) or len(entry) < 2:
            skipped += 1
            continue
        graph, _token = entry

        labels = stem_to_label.get(stem)
        if labels is None:
            skipped += 1
            continue

        try:
            batch = Batch.from_data_list([graph]).to(device)
            x = batch.x.float()
            edge_index = batch.edge_index
            batch_vec = batch.batch
            edge_attr = getattr(batch, "edge_attr", None)

            with torch.no_grad():
                result = gnn(
                    x, edge_index, batch_vec, edge_attr,
                    return_intermediates=True,
                )
                # Returns: (node_embs, batch, jk_entropy, intermediates)
                if len(result) == 4:
                    _, _, _, intermediates = result
                else:
                    log.warning(f"  Unexpected return length {len(result)} from GNNEncoder")
                    skipped += 1
                    continue

            # Pool each phase over function-level nodes
            node_types = (x[:, 0].float() * _MAX_TYPE_ID).round().long()
            func_mask = torch.isin(node_types, _FUNC_TYPE_TENSOR.to(device))

            for phase_key in _PHASE_KEYS:
                phase_emb = intermediates[phase_key]  # [N, 256]
                if func_mask.any():
                    func_embs = phase_emb[func_mask]               # [K, 256]
                    pooled = torch.cat([
                        func_embs.max(0).values,                   # [256] max
                        func_embs.mean(0),                         # [256] mean
                    ], dim=0)                                       # [512] — matches sentinel_model global_max+mean_pool
                else:
                    all_nodes = phase_emb
                    pooled = torch.cat([
                        all_nodes.max(0).values,
                        all_nodes.mean(0),
                    ], dim=0)
                phase_embs[phase_key].append(pooled.float().cpu().numpy())

            all_labels.append(labels)
            valid_stems.append(stem)

        except Exception as exc:
            log.debug(f"  Skipping {stem}: {exc}")
            skipped += 1
            continue

    log.info(f"  Extracted embeddings for {len(valid_stems)} contracts "
             f"({skipped} skipped)")

    if not valid_stems:
        raise RuntimeError("No embeddings extracted — check model/cache compatibility.")

    embeddings = {k: np.vstack(v) for k, v in phase_embs.items()}
    labels_arr = np.array(all_labels, dtype=np.int32)

    return embeddings, labels_arr, valid_stems


# ── Probing ───────────────────────────────────────────────────────────────────

def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_name: str,
) -> dict:
    """
    Train a logistic regression probe and evaluate on test set.

    Returns:
        dict with f1, auroc, n_positive_train, n_positive_test
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    # Handle degenerate case: no positive examples in train
    if y_train.sum() == 0:
        return {"f1": 0.0, "auroc": 0.5, "n_pos_train": 0, "n_pos_test": int(y_test.sum()),
                "note": "no_positive_train"}

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=1)
    try:
        clf.fit(X_train, y_train)
    except Exception as exc:
        return {"f1": 0.0, "auroc": 0.5, "error": str(exc),
                "n_pos_train": int(y_train.sum()), "n_pos_test": int(y_test.sum())}

    y_pred = clf.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # AUROC (requires positive examples in test)
    try:
        if y_test.sum() > 0 and (1 - y_test).sum() > 0:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(y_test, y_proba))
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5

    return {
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "n_pos_train": int(y_train.sum()),
        "n_pos_test": int(y_test.sum()),
    }


def run_all_probes(
    embeddings: dict[str, np.ndarray],  # phase_key → [N, 256]
    labels: np.ndarray,                  # [N, 10]
    test_frac: float = 0.20,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Train probes for all phases × all classes.

    Returns:
        {phase_key: [result_class0, result_class1, ..., result_class9]}
    """
    from sklearn.model_selection import train_test_split

    n = labels.shape[0]
    results: dict[str, list[dict]] = {}

    for phase_key in _PHASE_KEYS:
        X = embeddings[phase_key]  # [N, 256]
        phase_results = []

        for ci, cname in enumerate(CLASS_NAMES):
            y = labels[:, ci]  # [N] binary

            # Stratified split when possible (need ≥2 positive examples)
            n_pos = int(y.sum())
            if n_pos >= 2:
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_frac, random_state=seed, stratify=y
                    )
                except Exception:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_frac, random_state=seed
                    )
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_frac, random_state=seed
                )

            res = run_probe(X_tr, y_tr, X_te, y_te, cname)
            res["class"] = cname
            phase_results.append(res)

        results[phase_key] = phase_results

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_f1_table(probe_results: dict[str, list[dict]]) -> None:
    """Print a 3×10 F1 table to stdout."""
    header = f"{'Class':25s} | {'Phase1':8s} | {'Phase2':8s} | {'Phase3':8s} | {'Δ P2-P1':8s} | {'Δ P3-P1':8s}"
    log.info("\n" + "=" * len(header))
    log.info("PROBING CLASSIFIER F1 (3 × 10 table)")
    log.info("=" * len(header))
    log.info(header)
    log.info("-" * len(header))

    p1 = probe_results["after_phase1"]
    p2 = probe_results["after_phase2"]
    p3 = probe_results["after_phase3"]

    for ci in range(len(CLASS_NAMES)):
        f1_p1 = p1[ci]["f1"]
        f1_p2 = p2[ci]["f1"]
        f1_p3 = p3[ci]["f1"]
        delta_21 = f1_p2 - f1_p1
        delta_31 = f1_p3 - f1_p1
        log.info(
            f"  {CLASS_NAMES[ci]:23s} | {f1_p1:8.4f} | {f1_p2:8.4f} | {f1_p3:8.4f} "
            f"| {delta_21:+8.4f} | {delta_31:+8.4f}"
        )

    log.info("=" * len(header))


def evaluate_pass_criteria(
    probe_results: dict[str, list[dict]],
) -> dict:
    """
    Evaluate Reentrancy pass criterion and print diagnostic.
    """
    p1 = probe_results["after_phase1"]
    p2 = probe_results["after_phase2"]

    reent_idx = CLASS_NAMES.index("Reentrancy")
    f1_p1 = p1[reent_idx]["f1"]
    f1_p2 = p2[reent_idx]["f1"]
    delta = f1_p2 - f1_p1

    passes = delta >= 0.03
    verdict = "PASS" if passes else "FAIL"
    log.info(f"\n  REENTRANCY PASS CRITERION (Phase2 F1 > Phase1 F1 + 3pp):")
    log.info(f"    Phase1 F1 = {f1_p1:.4f}, Phase2 F1 = {f1_p2:.4f}, Δ = {delta:+.4f}")
    log.info(f"    Result: {verdict}")

    if not passes:
        log.warning(
            "  CFG/ICFG phase adds no linearly-separable signal for Reentrancy "
            "(Phase2 probe F1 ≈ Phase1 probe F1)"
        )

    return {
        "reentrancy_f1_phase1": round(f1_p1, 4),
        "reentrancy_f1_phase2": round(f1_p2, 4),
        "reentrancy_delta": round(delta, 4),
        "reentrancy_criterion_pass": passes,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probing classifiers per GNN phase for SENTINEL"
    )
    add_common_args(parser, require_checkpoint=True)
    # Note: --n-contracts is already added by add_common_args; set a different default here
    parser.set_defaults(n_contracts=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.out) if args.out else Path("ml/logs/interpretability/l5_probing_classifiers.json")
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        model = load_model(
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            phase2_edge_types=getattr(args, "phase2_edge_types", None),
        )
    except Exception as exc:
        log.error(f"Model load failed: {exc}")
        return 1

    # ── Load split ─────────────────────────────────────────────────────────────
    try:
        stems, df_split, cache = load_val_split(
            cache_path=Path(args.cache),
            label_csv=Path(args.label_csv),
            splits_dir=Path(args.splits_dir),
            split=getattr(args, "split", "val"),
        )
    except Exception as exc:
        log.error(f"Data load failed: {exc}")
        return 1

    # ── Extract embeddings ─────────────────────────────────────────────────────
    log.info(f"Extracting per-phase embeddings for up to {args.n_contracts} contracts...")
    try:
        embeddings, labels, valid_stems = extract_phase_embeddings(
            model=model,
            stems=stems,
            cache=cache,
            df_split=df_split,
            device=args.device,
            max_contracts=args.n_contracts,
            seed=getattr(args, "seed", 42),
        )
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    log.info(f"Embeddings shape: {embeddings['after_phase1'].shape}")
    log.info(f"Labels shape: {labels.shape}")
    log.info(f"Label prevalence: {labels.mean(axis=0).round(3).tolist()}")

    # ── Train probes ───────────────────────────────────────────────────────────
    log.info("Training probing classifiers (logistic regression)...")
    probe_results = run_all_probes(
        embeddings=embeddings,
        labels=labels,
        test_frac=0.20,
        seed=getattr(args, "seed", 42),
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_f1_table(probe_results)

    # ── Evaluate pass criteria ────────────────────────────────────────────────
    criteria = evaluate_pass_criteria(probe_results)

    # ── Heatmap of F1 values (3 phases × 10 classes) ─────────────────────────
    f1_matrix = np.array([
        [probe_results[pk][ci]["f1"] for ci in range(len(CLASS_NAMES))]
        for pk in _PHASE_KEYS
    ])  # [3, 10]
    try:
        plot_class_heatmap(
            matrix=f1_matrix,
            row_labels=[f"Phase{i+1}" for i in range(3)],
            col_labels=CLASS_NAMES,
            title="Probing Classifier F1 — Phase × Class",
            output_path=out_dir / "l5_probing_f1_heatmap.png",
            fmt=".3f",
            cmap="Blues",
            figsize=(14, 4),
        )
    except Exception as exc:
        log.warning(f"Heatmap failed: {exc}")

    # ── JSON output ───────────────────────────────────────────────────────────
    report = {
        "experiment": "exp_l5_probing_classifiers",
        "checkpoint": str(args.checkpoint),
        "n_contracts_used": len(valid_stems),
        "split": getattr(args, "split", "val"),
        "pass_criteria": criteria,
        "probe_results": {
            pk: probe_results[pk] for pk in _PHASE_KEYS
        },
        "phase_names": dict(zip(_PHASE_KEYS, PHASE_NAMES)),
    }
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nJSON report saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
